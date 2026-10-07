#!/usr/bin/env python3
"""Draw the retrieval-evaluation figure from the committed per-query results (500 random queries, 2026-10-01).

    pip install pandas matplotlib
    python docs/figures/make_figures.py        # writes docs/media/retrieval-eval.png

Input: docs/examples/retrieval-eval-2026-10-01.csv, a copy of the report written by
`python eval_retrieval.py --source qdrant --csv report.csv` (the original report*.csv files are git-ignored).
Intervals: Wilson 95 % for proportions; percentile bootstrap (5,000 resamples, seed 42) for MRR.
"""
import math
import random
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "media"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE, GREY, RED = "#5b6ee1", "#e07b39", "#9aa0a8", "#c0392b"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

df = pd.read_csv(ROOT / "docs" / "examples" / "retrieval-eval-2026-10-01.csv")
clean = df[~df["ambiguous_title"]]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def boot_mean(values, n_boot=5000, seed=42):
    rng = random.Random(seed)
    vals = list(values)
    boots = sorted(statistics.mean(rng.choices(vals, k=len(vals))) for _ in range(n_boot))
    return boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


metrics = {}
for name, part in (("all queries", df), ("excluding ambiguous titles", clean)):
    n = len(part)
    h1, h10 = int(part["hit@1"].sum()), int(part["hit@10"].sum())
    metrics[name] = {
        "n": n,
        "Hit@1": (h1 / n, *wilson(h1, n)),
        "Hit@10": (h10 / n, *wilson(h10, n)),
        "MRR": (part["reciprocal_rank"].mean(), *boot_mean(part["reciprocal_rank"])),
    }

fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))

# 1. the three metrics with intervals, all vs excluding ambiguous titles
names = ["Hit@1", "Hit@10", "MRR"]
w = 0.38
for j, (label, colour) in enumerate((("all queries", GREY), ("excluding ambiguous titles", BLUE))):
    m = metrics[label]
    vals = [m[k][0] for k in names]
    err = [[m[k][0] - m[k][1] for k in names], [m[k][2] - m[k][0] for k in names]]
    xs = [i + (j - 0.5) * w for i in range(3)]
    ax[0].bar(xs, vals, w, yerr=err, capsize=3, color=colour, label=f"{label} (n = {m['n']})")
    for x, v in zip(xs, vals):
        ax[0].text(x, v + 0.045, f"{v:.3f}", ha="center", fontsize=9)
ax[0].set_xticks(range(3), names)
ax[0].set_ylim(0, 1.22)
ax[0].set_title("Title as query, 95 % intervals")
ax[0].legend(loc="upper center", fontsize=8.5, frameon=False)

# 2. where the true series was ranked
order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
def dist(part):
    c = part["rank"].value_counts()
    return [int(c.get(float(r), 0)) for r in order] + [int(part["rank"].isna().sum())]

d_all, d_amb = dist(df), dist(df[df["ambiguous_title"]])
labels = [str(r) for r in order] + ["not in\ntop 10"]
ax[1].bar(labels, [a - b for a, b in zip(d_all, d_amb)], color=BLUE, label="unambiguous title")
ax[1].bar(labels, d_amb, bottom=[a - b for a, b in zip(d_all, d_amb)], color=ORANGE, label="title shared with another series")
ax[1].set_yscale("log")
ax[1].set_ylim(0.7, 800)
ax[1].set_xlabel("rank of the series itself")
ax[1].set_ylabel("queries (log scale)")
ax[1].set_title("Where the series ranks (n = 500)")
ax[1].legend(fontsize=8.5, frameon=False)

# 3. outcome per group
def outcome(part):
    n = len(part)
    first = int(part["hit@1"].sum())
    found = int(part["hit@10"].sum()) - first
    return [first / n, found / n, (n - first - found) / n]

groups = {"unambiguous\n(n = %d)" % len(clean): outcome(clean),
          "shared title\n(n = %d)" % int(df["ambiguous_title"].sum()): outcome(df[df["ambiguous_title"]])}
bottom = [0, 0]
for i, (lab, colour) in enumerate((("rank 1", BLUE), ("rank 2-10", GREY), ("not in top 10", RED))):
    vals = [g[i] for g in groups.values()]
    ax[2].bar(list(groups), vals, bottom=bottom, color=colour, label=lab)
    for x, (v, b) in enumerate(zip(vals, bottom)):
        if v > 0.04:
            ax[2].text(x, b + v / 2, f"{v:.0%}", ha="center", va="center", color="white", fontsize=10)
    bottom = [b + v for b, v in zip(bottom, vals)]
ax[2].set_ylim(0, 1)
ax[2].set_title("Outcome by title type")
ax[2].legend(fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)

fig.suptitle("Retrieval check: does the series come back for its own title? (500 random series, live index, 2026-10-01)", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "retrieval-eval.png", bbox_inches="tight")

for label, m in metrics.items():
    print(label, {k: tuple(round(x, 3) for x in v) for k, v in m.items() if k != "n"}, "n =", m["n"])

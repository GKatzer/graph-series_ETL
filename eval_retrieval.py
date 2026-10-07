"""
eval_retrieval.py
~~~~~~~~~~~~~~~~~
Оценивает качество семантического поиска в Qdrant на ground truth: берёт
случайную выборку (name, overview, tmdb_id), кодирует title (или первое
предложение summary) как поисковый запрос — тем же query-префиксом, что
использует inference/main.py в graph-series_ml — ищет в Qdrant и проверяет,
находится ли исходный tmdb_id среди результатов. Отдельный ручной скрипт
по аналогии с validate.py, только на уровне ранжирования эмбеддингов, а не
схемы Parquet. Backend не нужен.

Ground truth по умолчанию читается прямо из payload коллекции в Qdrant
через scroll (--source qdrant) — так эвал не зависит от того, успел ли
локально прогнаться весь tmdb_fetcher.py. --source parquet — из
data/raw/tmdb_series.parquet (tmdb_id / name / overview), если он уже
есть локально (быстрее, без сети до Qdrant для самой выборки).

Метрики: Hit@1, Hit@k, MRR.

Запуск:
  python eval_retrieval.py                    # 500 случайных запросов, отчёт в stdout
  python eval_retrieval.py --n 1000 --seed 7
  python eval_retrieval.py --query-field summary
  python eval_retrieval.py --source qdrant    # ground truth из Qdrant, а не Parquet
  python eval_retrieval.py --verbose          # + примеры промахов
  python eval_retrieval.py --csv report.csv   # сохранить построчный отчёт

Если ground truth (после фильтров, до .sample()) меньше MIN_POPULATION строк —
падаем с ошибкой: --source auto мог тихо подхватить локальный смоук-тестовый
Parquet вместо боевой коллекции. --allow-small снимает эту защиту осознанно.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import SearchRequest
from sentence_transformers import SentenceTransformer

load_dotenv()

log = logging.getLogger("eval_retrieval")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Настройки — как в .env / inference/main.py (graph-series_ml)
# ---------------------------------------------------------------------------
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION  = os.getenv("QDRANT_COLLECTION", "graph-series")

MODEL_NAME   = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

TEXTS_PATH = Path("data/raw/tmdb_series.parquet")  # tmdb_id / name / overview

# Payload коллекции 'graph-series' (qdrant_loader.py)
ID_COLUMN         = "tmdb_id"
PAYLOAD_TITLE_KEY = "name"
PAYLOAD_TEXT_KEY  = "overview"

DEFAULT_N       = 500
DEFAULT_TOP_K   = 10
MIN_POPULATION  = 1000  # ниже этого — похоже на случайно подхваченный смоук-тестовый Parquet, а не боевые данные
SEARCH_BATCH    = 100   # запросов за один search_batch
SCROLL_BATCH    = 1000  # точек за один scroll при чтении ground truth из Qdrant


# ---------------------------------------------------------------------------
# Ground truth из Qdrant (когда series_texts.parquet отсутствует)
# ---------------------------------------------------------------------------
def load_ground_truth_from_qdrant(client: QdrantClient) -> pd.DataFrame:
    log.info("Тянем ground truth прямо из payload коллекции '%s'...", COLLECTION)
    rows = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            with_payload=True,
            with_vectors=False,
            limit=SCROLL_BATCH,
            offset=offset,
        )
        rows.extend({
            ID_COLUMN: p.payload.get(ID_COLUMN),
            "title":   p.payload.get(PAYLOAD_TITLE_KEY, ""),
            "summary": p.payload.get(PAYLOAD_TEXT_KEY, ""),
        } for p in points)
        if offset is None:
            break
    log.info("Точек в коллекции '%s': %d", COLLECTION, len(rows))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Построение запроса
# ---------------------------------------------------------------------------
def first_sentence(text: str) -> str:
    text = text.strip()
    match = re.search(r"(.+?[.!?])(\s|$)", text)
    return match.group(1) if match else text


def build_query(row: pd.Series, field: str) -> str:
    if field == "title":
        return str(row["title"]).strip()
    return first_sentence(str(row["summary"]))


# ---------------------------------------------------------------------------
# Поиск в Qdrant + метрики
# ---------------------------------------------------------------------------
def search_all(client: QdrantClient, vectors, top_k: int) -> list[list]:
    hits: list[list] = []
    for i in range(0, len(vectors), SEARCH_BATCH):
        chunk = vectors[i : i + SEARCH_BATCH]
        requests = [
            SearchRequest(
                vector=[float(x) for x in vec],
                limit=top_k,
                with_payload=True,
            )
            for vec in chunk
        ]
        hits.extend(client.search_batch(collection_name=COLLECTION, requests=requests))
    return hits


def evaluate(sample: pd.DataFrame, queries: list[str], hits: list[list], top_k: int) -> pd.DataFrame:
    rows = []
    for (_, row), query, result in zip(sample.iterrows(), queries, hits):
        true_id = row[ID_COLUMN]
        rank = next(
            (pos for pos, h in enumerate(result, start=1) if h.payload.get(ID_COLUMN) == true_id),
            None,
        )
        rows.append({
            ID_COLUMN: true_id,
            "query": query,
            "title": row["title"],
            "ambiguous_title": row["ambiguous_title"],
            "rank": rank,
            "reciprocal_rank": 1.0 / rank if rank else 0.0,
            "hit@1": rank == 1,
            f"hit@{top_k}": bool(rank),
            "top1_title": result[0].payload.get(PAYLOAD_TITLE_KEY) if result else None,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Печать отчёта — в стиле validate.py
# ---------------------------------------------------------------------------
RESET  = "\033[0m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"


def _color(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


def print_report(df: pd.DataFrame, top_k: int, query_field: str, verbose: bool) -> None:
    n = len(df)
    n_ambiguous = int(df["ambiguous_title"].sum())
    clean = df[~df["ambiguous_title"]]
    n_clean = len(clean)

    def _fmt(series: pd.Series) -> str:
        return f"{series.mean():.3f}" if len(series) else "—"

    print()
    print(_color("═" * 60, BOLD))
    print(_color("  Retrieval Eval Report (Qdrant)", BOLD))
    print(_color("═" * 60, BOLD))
    print(f"\n  коллекция: {_color(COLLECTION, CYAN)}  |  запросов: {n}  "
          f"|  поле запроса: {query_field}  |  top-k: {top_k}")
    print(f"  неоднозначных названий (есть тёзка в каталоге): {n_ambiguous} / {n} "
          f"({n_ambiguous / n * 100:.1f}%)\n")

    print(f"  {_color('Hit@1', CYAN)}   (все, as-is):                  {_fmt(df['hit@1'])}")
    print(f"  {_color('Hit@1', CYAN)}   (без неоднозначных названий):  {_fmt(clean['hit@1'])}  "
          f"N={n_clean}, исключено {n_ambiguous} ({n_ambiguous / n * 100:.1f}%)")
    print(f"  {_color(f'Hit@{top_k}', CYAN)}  (все, as-is):                  {_fmt(df[f'hit@{top_k}'])}")
    print(f"  {_color(f'Hit@{top_k}', CYAN)}  (без неоднозначных названий):  {_fmt(clean[f'hit@{top_k}'])}")
    print(f"  {_color('MRR', CYAN)}     (все, as-is):                  {_fmt(df['reciprocal_rank'])}")
    print(f"  {_color('MRR', CYAN)}     (без неоднозначных названий):  {_fmt(clean['reciprocal_rank'])}")

    if verbose:
        ambiguous_rows = df[df["ambiguous_title"]]
        print(f"\n{_color(f'Неоднозначные названия (есть тёзка в каталоге): '
                           f'{len(ambiguous_rows)} / {n}', YELLOW)}")
        for _, r in ambiguous_rows.head(10).iterrows():
            rank_str = f"rank={int(r['rank'])}" if pd.notna(r["rank"]) else f"не найден в топ-{top_k}"
            print(f"    запрос:   {r['query']!r}  ({rank_str})")
            print(f"      ожидали: {r['title']!r}  ({r[ID_COLUMN]})")
            print(f"      топ-1:   {r['top1_title']!r}")

        real_misses = df[~df["ambiguous_title"] & df["rank"].isna()]
        print(f"\n{_color(f'Настоящие промахи (без неоднозначных названий, не найдено в топ-{top_k}): '
                           f'{len(real_misses)} / {n}', YELLOW)}")
        for _, r in real_misses.head(10).iterrows():
            print(f"    запрос:   {r['query']!r}")
            print(f"      ожидали: {r['title']!r}  ({r[ID_COLUMN]})")
            print(f"      топ-1:   {r['top1_title']!r}")

    print()
    print(_color("─" * 60, BOLD))
    print()


# ---------------------------------------------------------------------------
# Основная логика
# ---------------------------------------------------------------------------
def run(n: int, seed: int, top_k: int, query_field: str, device: str,
        source: str, verbose: bool, csv_path: Path | None, allow_small: bool = False) -> None:
    log.info("Подключаемся к Qdrant: %s:%d (HTTP), коллекция '%s'",
             QDRANT_HOST, QDRANT_PORT, COLLECTION)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)

    use_qdrant_source = source == "qdrant" or (source == "auto" and not TEXTS_PATH.exists())
    if use_qdrant_source:
        df = load_ground_truth_from_qdrant(client)
    else:
        if not TEXTS_PATH.exists():
            log.error("Не найден %s — сначала прогони tmdb_fetcher.py --details, "
                       "либо запусти с --source qdrant", TEXTS_PATH)
            raise SystemExit(1)
        log.info("Загружаем %s...", TEXTS_PATH)
        df = pd.read_parquet(TEXTS_PATH)[[ID_COLUMN, PAYLOAD_TITLE_KEY, PAYLOAD_TEXT_KEY]]
        df = df.rename(columns={PAYLOAD_TITLE_KEY: "title", PAYLOAD_TEXT_KEY: "summary"})
    df = df.dropna(subset=[ID_COLUMN, "title", "summary"])
    df = df[df["title"].str.strip().astype(bool)]
    if query_field == "summary":
        df = df[df["summary"].str.strip().astype(bool)]

    # Неоднозначные названия — считаем по всему каталогу (df), не по выборке:
    # конкурирующий тёзка почти никогда не попадёт в сами 500 сэмплированных
    # строк, он сидит в индексе Qdrant как один из кандидатов при поиске.
    dup_titles = set(df.loc[df["title"].duplicated(keep=False), "title"])

    if len(df) < MIN_POPULATION and not allow_small:
        source_desc = f"коллекции '{COLLECTION}'" if use_qdrant_source else str(TEXTS_PATH)
        log.error(
            "Ground truth подозрительно мал: %d строк из %s (ожидали хотя бы %d). "
            "Похоже, --source auto подхватил локальный смоук-тестовый Parquet вместо "
            "боевой коллекции. Запусти с --source qdrant, чтобы взять полную коллекцию, "
            "или --allow-small, если маленькая выборка — осознанный выбор.",
            len(df), source_desc, MIN_POPULATION,
        )
        raise SystemExit(1)

    sample_n = min(n, len(df))
    if sample_n < n:
        source_desc = f"коллекции '{COLLECTION}'" if use_qdrant_source else str(TEXTS_PATH)
        log.warning("В %s только %d пригодных строк — беру все", source_desc, sample_n)
    sample = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)
    sample["ambiguous_title"] = sample["title"].isin(dup_titles)
    log.info("Строк в выборке: %d  |  с неоднозначным названием: %d",
             sample_n, int(sample["ambiguous_title"].sum()))

    queries = [build_query(row, query_field) for _, row in sample.iterrows()]
    prefixed = [q if q.startswith(QUERY_PREFIX) else QUERY_PREFIX + q for q in queries]

    log.info("Загружаем модель %s (device=%s)...", MODEL_NAME, device)
    model = SentenceTransformer(MODEL_NAME, device=device)

    log.info("Кодируем %d запросов...", len(prefixed))
    vectors = model.encode(
        prefixed,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    log.info("Ищем в Qdrant (top-%d, батчами по %d)...", top_k, SEARCH_BATCH)
    t0 = time.time()
    hits = search_all(client, vectors, top_k)
    elapsed = time.time() - t0
    log.info("✔ Поиск завершён за %.1f сек (%.0f q/s)", elapsed, len(hits) / elapsed if elapsed else 0)

    report_df = evaluate(sample, queries, hits, top_k)
    print_report(report_df, top_k, query_field, verbose)

    if csv_path:
        report_df.to_csv(csv_path, index=False)
        log.info("✔ Построчный отчёт сохранён: %s", csv_path)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка качества ретривала эмбеддингов в Qdrant")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                         help=f"Размер случайной выборки (по умолчанию {DEFAULT_N})")
    parser.add_argument("--seed", type=int, default=42, help="Random seed для выборки")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K,
                         help=f"Глубина поиска для Hit@k (по умолчанию {DEFAULT_TOP_K})")
    parser.add_argument("--query-field", choices=["title", "summary"], default="title",
                         help="Из чего строить запрос: title или первое предложение summary")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--source", choices=["auto", "parquet", "qdrant"], default="auto",
                         help="Откуда брать ground truth: series_texts.parquet, payload "
                              "коллекции в Qdrant, или auto (parquet если есть, иначе qdrant)")
    parser.add_argument("--verbose", action="store_true", help="Показать примеры промахов")
    parser.add_argument("--csv", type=Path, default=None, help="Сохранить построчный отчёт в CSV")
    parser.add_argument("--allow-small", action="store_true",
                         help=f"Не падать, если ground truth меньше {MIN_POPULATION} строк "
                              "(иначе это считается признаком случайно подхваченного смоук-теста)")
    args = parser.parse_args()

    run(
        n=args.n,
        seed=args.seed,
        top_k=args.k,
        query_field=args.query_field,
        device=args.device,
        source=args.source,
        verbose=args.verbose,
        csv_path=args.csv,
        allow_small=args.allow_small,
    )


if __name__ == "__main__":
    main()

"""
embeddings.py
~~~~~~~~~~~~~
Считает эмбеддинги описаний сериалов на GPU и сохраняет в Parquet.
Модель: BAAI/bge-small-en-v1.5 (384 измерения).

Вход:  data/raw/tmdb_series.parquet         (tmdb_id, name, overview, ...)
       data/raw/tmdb_keywords.parquet       (series_id, keyword_name)
Выход: data/processed/tmdb_embeddings.parquet (tmdb_id, embedding)

Запуск:
  python embeddings.py
  python embeddings.py --batch-size 256   # для GPU с меньшим VRAM
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

log = logging.getLogger("embeddings")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------
MODEL_NAME  = "BAAI/bge-small-en-v1.5"
BATCH_SIZE  = 512       # для RTX 2060 Super (8GB VRAM) — ок; уменьши если OOM
DEVICE      = "cuda"    # "cpu" если нет GPU
CHECKPOINT_EVERY = 20_000   # текстов между записями на диск

SERIES_PATH   = Path("data/raw/tmdb_series.parquet")
KEYWORDS_PATH = Path("data/raw/tmdb_keywords.parquet")
OUTPUT_DIR    = Path("data/processed")
OUTPUT_PATH   = OUTPUT_DIR / "tmdb_embeddings.parquet"


# ---------------------------------------------------------------------------
# Подготовка текста
# ---------------------------------------------------------------------------
def prepare_text(row: pd.Series) -> str:
    """
    bge-модели лучше работают с prefix 'Represent this sentence for searching relevant passages:'
    для query-эмбеддингов. Для документов (наш случай, graph-series_backend/embedder.py
    придерживается той же конвенции) — без префикса.
    """
    name     = str(row["name"]).strip()
    overview = str(row["overview"]).strip()
    text = f"{name}. {overview}"
    if row["keywords"]:
        text += f" Keywords: {row['keywords']}"
    return text


# ---------------------------------------------------------------------------
# Основная логика
# ---------------------------------------------------------------------------
def run(batch_size: int = BATCH_SIZE, device: str = DEVICE) -> None:
    log.info("Загружаем тексты из %s...", SERIES_PATH)
    df = pd.read_parquet(SERIES_PATH)
    df = df.dropna(subset=["overview"])
    df = df[df["overview"].str.strip().astype(bool)]

    log.info("Подтягиваем keywords из %s...", KEYWORDS_PATH)
    kw = pd.read_parquet(KEYWORDS_PATH)
    kw_joined = (
        kw.groupby("series_id")["keyword_name"]
        .apply(lambda names: ", ".join(names))
        .rename("keywords")
    )
    df = df.join(kw_joined, on="tmdb_id")
    df["keywords"] = df["keywords"].fillna("")

    # Пропускаем уже посчитанные
    if OUTPUT_PATH.exists():
        done = pd.read_parquet(OUTPUT_PATH)["tmdb_id"].tolist()
        df = df[~df["tmdb_id"].isin(done)]
        log.info("Уже посчитано: %d  |  Осталось: %d", len(done), len(df))
    else:
        log.info("Серий для эмбеддинга: %d", len(df))

    if df.empty:
        log.info("Всё уже посчитано.")
        return

    total = len(df)

    log.info("Загружаем модель: %s (device=%s)", MODEL_NAME, device)
    model = SentenceTransformer(MODEL_NAME, device=device)

    texts = df.apply(prepare_text, axis=1).tolist()
    ids = df["tmdb_id"].values

    log.info("Считаем эмбеддинги (batch_size=%d, порциями по %d)...", batch_size, CHECKPOINT_EVERY)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Порциями с записью на диск: оборванный прогон теряет максимум одну порцию,
    # а перезапуск пропускает уже посчитанные tmdb_id (см. выше).
    for start in range(0, total, CHECKPOINT_EVERY):
        embeddings = model.encode(
            texts[start:start + CHECKPOINT_EVERY],
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,   # L2-нормализация — нужна для cosine similarity
            convert_to_numpy=True,
        )
        new_df = pd.DataFrame({
            "tmdb_id":   ids[start:start + CHECKPOINT_EVERY],
            "embedding": list(embeddings.astype(np.float32)),
        })
        if OUTPUT_PATH.exists():
            new_df = pd.concat([pd.read_parquet(OUTPUT_PATH), new_df], ignore_index=True)
        new_df.to_parquet(OUTPUT_PATH, index=False, compression="zstd")
        log.info("  Сохранено %d / %d", min(start + CHECKPOINT_EVERY, total), total)

    elapsed = time.time() - t0
    log.info("✔ Готово за %.1f сек (%.0f vec/s) → %s (%.1f MB)",
             elapsed, total / elapsed, OUTPUT_PATH, OUTPUT_PATH.stat().st_size / 1e6)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Compute BGE embeddings for series")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=DEVICE, choices=["cuda", "cpu"])
    args = parser.parse_args()

    run(batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()
"""
qdrant_loader.py
~~~~~~~~~~~~~~~~
Загружает эмбеддинги в Qdrant через bulk upsert.

Point id = tmdb_id (int) напрямую — graph-series_backend делает
retrieve(ids=[tmdb_id]), так что id точки должен совпадать с tmdb_id,
а не быть автоинкрементным счётчиком.

Коллекцию создаёт graph-series_ml/scripts/init_qdrant.py (HNSW,
int8-квантизация, payload-индексы на tmdb_id/name) — --recreate здесь
делать не нужно в обычном режиме, только если осознанно пересоздаёшь
коллекцию заново.

Вход:  data/processed/tmdb_embeddings.parquet  (tmdb_id, embedding)
       data/raw/tmdb_series.parquet            (tmdb_id, name, overview, imdb_id, start_year)
Выход: коллекция 'graph-series' в Qdrant на VDS2

Запуск:
  python qdrant_loader.py
  python qdrant_loader.py --recreate   # пересоздать коллекцию с нуля (осторожно!)
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)

load_dotenv()

log = logging.getLogger("qdrant_loader")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Настройки — из .env
# ---------------------------------------------------------------------------
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION  = os.getenv("QDRANT_COLLECTION", "graph-series")

VECTOR_SIZE = 384    # bge-small-en-v1.5
BATCH_SIZE  = 256    # точек за один upsert

INPUT_PATH  = Path("data/processed/tmdb_embeddings.parquet")
SERIES_PATH = Path("data/raw/tmdb_series.parquet")


# ---------------------------------------------------------------------------
# Создание коллекции
# ---------------------------------------------------------------------------
def create_collection(client: QdrantClient) -> None:
    # Удаляем если существует
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)
        log.info("Коллекция '%s' удалена", COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
        hnsw_config=HnswConfigDiff(
            m=16,
            ef_construct=100,
        ),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            )
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20_000,
        ),
    )
    log.info("✔ Коллекция '%s' создана (HNSW + INT8 quantization)", COLLECTION)


# ---------------------------------------------------------------------------
# Загрузка
# ---------------------------------------------------------------------------
def run(recreate: bool = False) -> None:
    log.info("Загружаем эмбеддинги из %s...", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)

    # Подтягиваем name/overview/imdb_id/start_year/overview_source для payload
    log.info("Загружаем тексты из %s...", SERIES_PATH)
    series = pd.read_parquet(SERIES_PATH)[
        ["tmdb_id", "name", "overview", "imdb_id", "start_year", "overview_source"]
    ]
    df = df.merge(series, on="tmdb_id", how="left")
    df["name"]            = df["name"].fillna("")
    df["overview"]        = df["overview"].fillna("")
    df["imdb_id"]         = df["imdb_id"].fillna("")
    df["overview_source"] = df["overview_source"].fillna("tmdb")

    total = len(df)
    log.info("Точек для загрузки: %d", total)

    log.info("Подключаемся к Qdrant: %s:%d (gRPC)", QDRANT_HOST, QDRANT_PORT + 1)
    client = QdrantClient(
        host=QDRANT_HOST,
        grpc_port=QDRANT_PORT + 1,   # 6334
        prefer_grpc=True,
        timeout=60,
        grpc_options={"grpc.enable_http_proxy": 0},
    )

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing or recreate:
        create_collection(client)
    else:
        log.info("Коллекция '%s' уже существует, дополняем", COLLECTION)

    t0 = time.time()
    uploaded = 0

    for i in range(0, total, BATCH_SIZE):
        batch = df.iloc[i : i + BATCH_SIZE]
        points = [
            PointStruct(
                id=int(row["tmdb_id"]),
                vector=[float(x) for x in row["embedding"]],
                payload={
                    "tmdb_id":         int(row["tmdb_id"]),
                    "name":            row["name"],
                    "overview":        row["overview"],
                    "overview_source": row["overview_source"],
                    "imdb_id":         row["imdb_id"],
                    "start_year":      None if pd.isna(row["start_year"]) else int(row["start_year"]),
                },
            )
            for _, row in batch.iterrows()
        ]
        client.upsert(collection_name=COLLECTION, points=points)
        uploaded += len(points)

        if uploaded % 5000 == 0 or uploaded == total:
            elapsed = time.time() - t0
            rps = uploaded / elapsed
            eta = (total - uploaded) / rps if rps > 0 else 0
            log.info(
                "  Прогресс: %d / %d  |  %.0f pts/s  |  ETA: %.0f сек",
                uploaded, total, rps, eta,
            )

    elapsed = time.time() - t0
    log.info("✔ Загружено %d точек за %.1f сек", total, elapsed)

    info = client.get_collection(COLLECTION)
    log.info("Коллекция '%s': %d точек, статус: %s",
             COLLECTION, info.points_count, info.status)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Load embeddings into Qdrant")
    parser.add_argument("--recreate", action="store_true",
                        help="Пересоздать коллекцию с нуля")
    args = parser.parse_args()
    run(recreate=args.recreate)


if __name__ == "__main__":
    main()
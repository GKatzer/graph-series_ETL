"""
neo4j_loader.py
~~~~~~~~~~~~~~~
Загружает все Parquet-файлы в Neo4j через batch MERGE-операции.

Схема — контракт graph-series_backend (scripts/schema_init.cypher,
routers/*.py): Series ключуется по tmdb_id, рёбра без свойств,
Person/Creators/Directors → Series, Series → Genre/Keyword/Country/
Language/Network. Индексы/constraints грузит не этот скрипт, а
schema_init.cypher в graph-series_backend — прогони его (заново) после
полной перезагрузки графа.

На сериал берём не больше MAX_CAST актёров (по billing order) и MAX_DIRECTORS
режиссёров (по числу эпизодов) — как на странице Methodology веб-приложения;
в raw-файлах остаётся всё. Лимиты меняются флагами --max-cast/--max-directors.

В граф загружаются только сериалы с vote_count >= 2 (~56k из ~211k
в Qdrant) — та же идея, что раньше давал top_series_ids.parquet, только
теперь фильтр по vote_count берётся прямо из tmdb_series.parquet.

Порядок загрузки важен — сначала узлы, потом рёбра:
  1. Series    ← tmdb_series.parquet (только vote_count >= 2)
  2. Person    ← cast + creators + directors (union по person_id)
  3. Genre / Keyword / Network / Country / Language
  4. Рёбра     ← все relation-файлы (только для series в графе)

Запуск:
  python neo4j_loader.py                    # всё
  python neo4j_loader.py --only nodes       # только узлы
  python neo4j_loader.py --only edges       # только рёбра
  python neo4j_loader.py --min-votes 5      # другой порог фильтра графа
  python neo4j_loader.py --max-cast 50 --max-directors 20   # другие лимиты на сериал
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

log = logging.getLogger("neo4j_loader")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Настройки подключения — из .env
# ---------------------------------------------------------------------------
NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
BATCH_SIZE     = 1000   # строк за одну транзакцию
MIN_VOTE_COUNT = 2      # порог фильтра графа — как в graph-series_backend
MAX_CAST       = 20     # актёров на сериал (по billing order); 0 = без лимита
MAX_DIRECTORS  = 10     # режиссёров на сериал (по episode_count); 0 = без лимита

# ---------------------------------------------------------------------------
# Пути к файлам
# ---------------------------------------------------------------------------
RAW = Path("data/raw")

# ---------------------------------------------------------------------------
# Cypher-запросы — узлы
# ---------------------------------------------------------------------------
MERGE_SERIES = """
UNWIND $rows AS row
MERGE (s:Series {tmdb_id: row.tmdb_id})
SET s.name = row.name,
    s.original_name = row.original_name,
    s.imdb_id = row.imdb_id,
    s.wikidata_id = row.wikidata_id,
    s.tvdb_id = toInteger(row.tvdb_id),
    s.start_year = toInteger(row.start_year),
    s.end_year = toInteger(row.end_year),
    s.status = row.status,
    s.type = row.type,
    s.in_production = row.in_production,
    s.original_language = row.original_language,
    s.season_count = toInteger(row.season_count),
    s.episode_count = toInteger(row.episode_count),
    s.episode_runtime = toFloat(row.episode_runtime),
    s.popularity = toFloat(row.popularity),
    s.vote_average = toFloat(row.vote_average),
    s.vote_count = toInteger(row.vote_count),
    s.us_content_rating = row.us_content_rating,
    s.overview = row.overview,
    s.tagline = row.tagline,
    s.homepage = row.homepage,
    s.poster_path = row.poster_path
"""

MERGE_PERSON = """
UNWIND $rows AS row
MERGE (p:Person {person_id: row.person_id})
SET p.name = row.person_name
"""

MERGE_GENRE = """
UNWIND $rows AS row
MERGE (g:Genre {genre_id: row.genre_id})
SET g.name = row.genre_name
"""

MERGE_KEYWORD = """
UNWIND $rows AS row
MERGE (k:Keyword {keyword_id: row.keyword_id})
SET k.name = row.keyword_name
"""

MERGE_NETWORK = """
UNWIND $rows AS row
MERGE (n:Network {network_id: row.network_id})
SET n.name = row.network_name,
    n.country = row.country
"""

MERGE_COUNTRY = """
UNWIND $rows AS row
MERGE (c:Country {code: row.code})
"""

MERGE_LANGUAGE = """
UNWIND $rows AS row
MERGE (l:Language {code: row.code})
"""

# --- Рёбра (без свойств — контракт backend их не читает) ---
MERGE_ACTED_IN = """
UNWIND $rows AS row
MATCH (p:Person {person_id: row.person_id})
MATCH (s:Series {tmdb_id: row.series_id})
MERGE (p)-[:ACTED_IN]->(s)
"""

MERGE_CREATED = """
UNWIND $rows AS row
MATCH (p:Person {person_id: row.person_id})
MATCH (s:Series {tmdb_id: row.series_id})
MERGE (p)-[:CREATED]->(s)
"""

MERGE_DIRECTED = """
UNWIND $rows AS row
MATCH (p:Person {person_id: row.person_id})
MATCH (s:Series {tmdb_id: row.series_id})
MERGE (p)-[:DIRECTED]->(s)
"""

MERGE_HAS_GENRE = """
UNWIND $rows AS row
MATCH (s:Series {tmdb_id: row.series_id})
MATCH (g:Genre {genre_id: row.genre_id})
MERGE (s)-[:HAS_GENRE]->(g)
"""

MERGE_HAS_KEYWORD = """
UNWIND $rows AS row
MATCH (s:Series {tmdb_id: row.series_id})
MATCH (k:Keyword {keyword_id: row.keyword_id})
MERGE (s)-[:HAS_KEYWORD]->(k)
"""

MERGE_PRODUCED_IN = """
UNWIND $rows AS row
MATCH (s:Series {tmdb_id: row.series_id})
MATCH (c:Country {code: row.code})
MERGE (s)-[:PRODUCED_IN]->(c)
"""

MERGE_HAS_LANGUAGE = """
UNWIND $rows AS row
MATCH (s:Series {tmdb_id: row.series_id})
MATCH (l:Language {code: row.code})
MERGE (s)-[:HAS_LANGUAGE]->(l)
"""

MERGE_AIRED_ON = """
UNWIND $rows AS row
MATCH (s:Series {tmdb_id: row.series_id})
MATCH (n:Network {network_id: row.network_id})
MERGE (s)-[:AIRED_ON]->(n)
"""


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def _batches(df: pd.DataFrame, size: int):
    for i in range(0, len(df), size):
        yield df.iloc[i : i + size].to_dict("records")


def run_batches(driver, cypher: str, df: pd.DataFrame, label: str) -> None:
    if df.empty:
        log.warning("⚠ Пустой DataFrame для '%s', пропускаем", label)
        return

    total = len(df)
    t0 = time.time()

    with driver.session() as session:
        for batch in _batches(df, BATCH_SIZE):
            session.run(cypher, rows=batch)

    elapsed = time.time() - t0
    log.info("  ✔ %s: %d строк за %.1f сек", label, total, elapsed)


def _load_series_ids(min_votes: int) -> list[int]:
    """tmdb_id сериалов, попадающих в граф (vote_count >= min_votes)."""
    df = pd.read_parquet(RAW / "tmdb_series.parquet")
    return df[df["vote_count"].fillna(0) >= min_votes]["tmdb_id"].tolist()


def _cap_per_series(df: pd.DataFrame, fname: str, max_cast: int, max_directors: int) -> pd.DataFrame:
    if fname == "tmdb_cast.parquet" and max_cast > 0:
        df = df.sort_values("order", na_position="last", kind="stable")
        return df.groupby("series_id", sort=False).head(max_cast)
    if fname == "tmdb_directors.parquet" and max_directors > 0:
        df = df.sort_values("episode_count", ascending=False, kind="stable")
        return df.groupby("series_id", sort=False).head(max_directors)
    return df


def _filtered(fname: str, series_ids: list[int], id_col: str = "series_id",
              max_cast: int = MAX_CAST, max_directors: int = MAX_DIRECTORS) -> pd.DataFrame:
    p = RAW / fname
    if not p.exists():
        log.warning("  ⚠ %s не найден, пропускаем", fname)
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df = df[df[id_col].isin(series_ids)]
    df = _cap_per_series(df, fname, max_cast, max_directors)
    return df.fillna("")


# ---------------------------------------------------------------------------
# Загрузка узлов
# ---------------------------------------------------------------------------
def load_nodes(driver, series_ids: list[int], max_cast: int = MAX_CAST, max_directors: int = MAX_DIRECTORS) -> None:
    log.info("═══ Загружаем узлы (%d сериалов в графе) ═══", len(series_ids))

    # Series
    series = pd.read_parquet(RAW / "tmdb_series.parquet")
    series = series[series["tmdb_id"].isin(series_ids)].fillna("")
    run_batches(driver, MERGE_SERIES, series, "Series")

    # Person — объединяем cast + creators + directors
    dfs = []
    for fname in ["tmdb_cast.parquet", "tmdb_creators.parquet", "tmdb_directors.parquet"]:
        df = _filtered(fname, series_ids, max_cast=max_cast, max_directors=max_directors)
        if not df.empty:
            dfs.append(df[["person_id", "person_name"]])
    if dfs:
        persons = pd.concat(dfs).drop_duplicates(subset="person_id")
        run_batches(driver, MERGE_PERSON, persons, "Person")

    # Genre
    df = _filtered("tmdb_genres.parquet", series_ids)
    if not df.empty:
        run_batches(driver, MERGE_GENRE, df[["genre_id", "genre_name"]].drop_duplicates(), "Genre")

    # Keyword
    df = _filtered("tmdb_keywords.parquet", series_ids)
    if not df.empty:
        run_batches(driver, MERGE_KEYWORD, df[["keyword_id", "keyword_name"]].drop_duplicates(), "Keyword")

    # Network
    df = _filtered("tmdb_networks.parquet", series_ids)
    if not df.empty:
        run_batches(driver, MERGE_NETWORK, df[["network_id", "network_name", "country"]].drop_duplicates(), "Network")

    # Country
    df = _filtered("tmdb_countries.parquet", series_ids)
    if not df.empty:
        run_batches(driver, MERGE_COUNTRY, df[["code"]].drop_duplicates(), "Country")

    # Language
    df = _filtered("tmdb_languages.parquet", series_ids)
    if not df.empty:
        run_batches(driver, MERGE_LANGUAGE, df[["code"]].drop_duplicates(), "Language")


# ---------------------------------------------------------------------------
# Загрузка рёбер
# ---------------------------------------------------------------------------
def load_edges(driver, series_ids: list[int], max_cast: int = MAX_CAST, max_directors: int = MAX_DIRECTORS) -> None:
    log.info("═══ Загружаем рёбра ═══")

    edge_map = [
        ("tmdb_cast.parquet",      MERGE_ACTED_IN,     "ACTED_IN"),
        ("tmdb_creators.parquet",  MERGE_CREATED,       "CREATED"),
        ("tmdb_directors.parquet", MERGE_DIRECTED,      "DIRECTED"),
        ("tmdb_genres.parquet",    MERGE_HAS_GENRE,     "HAS_GENRE"),
        ("tmdb_keywords.parquet",  MERGE_HAS_KEYWORD,   "HAS_KEYWORD"),
        ("tmdb_countries.parquet", MERGE_PRODUCED_IN,   "PRODUCED_IN"),
        ("tmdb_languages.parquet", MERGE_HAS_LANGUAGE,  "HAS_LANGUAGE"),
        ("tmdb_networks.parquet",  MERGE_AIRED_ON,      "AIRED_ON"),
    ]

    for fname, cypher, label in edge_map:
        df = _filtered(fname, series_ids, max_cast=max_cast, max_directors=max_directors)
        run_batches(driver, cypher, df, label)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------
def run(only: str | None = None, min_votes: int = MIN_VOTE_COUNT,
        max_cast: int = MAX_CAST, max_directors: int = MAX_DIRECTORS) -> None:
    series_ids = _load_series_ids(min_votes)
    log.info("Сериалов с vote_count >= %d: %d", min_votes, len(series_ids))

    log.info("Подключаемся к Neo4j: %s", NEO4J_URI)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        driver.verify_connectivity()
        log.info("✔ Соединение установлено")

        t0 = time.time()

        if only is None:
            load_nodes(driver, series_ids, max_cast, max_directors)
            load_edges(driver, series_ids, max_cast, max_directors)
        elif only == "nodes":
            load_nodes(driver, series_ids, max_cast, max_directors)
        elif only == "edges":
            load_edges(driver, series_ids, max_cast, max_directors)

        log.info("✔ Загрузка завершена за %.1f сек", time.time() - t0)
        log.info("Не забудь (пере)прогнать schema_init.cypher из graph-series_backend, "
                 "если это был полный wipe+reload")

    finally:
        driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Load TMDB Parquet data into Neo4j")
    parser.add_argument("--only", choices=["nodes", "edges"], default=None,
                         help="Загрузить только указанную часть (по умолчанию — всё)")
    parser.add_argument("--min-votes", type=int, default=MIN_VOTE_COUNT,
                         help=f"Порог vote_count для попадания в граф (default {MIN_VOTE_COUNT})")
    parser.add_argument("--max-cast", type=int, default=MAX_CAST,
                         help=f"Актёров на сериал, 0 = без лимита (default {MAX_CAST})")
    parser.add_argument("--max-directors", type=int, default=MAX_DIRECTORS,
                         help=f"Режиссёров на сериал, 0 = без лимита (default {MAX_DIRECTORS})")
    args = parser.parse_args()
    run(only=args.only, min_votes=args.min_votes, max_cast=args.max_cast, max_directors=args.max_directors)


if __name__ == "__main__":
    main()

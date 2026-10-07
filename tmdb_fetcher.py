"""
tmdb_fetcher.py
~~~~~~~~~~~~~~~
Выгружает данные о сериалах из TMDB API. Два шага:

  discover — скачивает daily ID export (все известные tmdb_id сериалов)
             → data/raw/tmdb_series_ids.parquet
  details  — по каждому tmdb_id один комбинированный запрос
             GET /tv/{id}?append_to_response=external_ids,keywords,
                          aggregate_credits,content_ratings
             → data/raw/tmdb_series.parquet       (свойства Series)
               data/raw/tmdb_genres.parquet        (series_id, genre_id, genre_name)
               data/raw/tmdb_keywords.parquet      (series_id, keyword_id, keyword_name)
               data/raw/tmdb_networks.parquet      (series_id, network_id, network_name, country)
               data/raw/tmdb_countries.parquet     (series_id, code)
               data/raw/tmdb_languages.parquet     (series_id, code)
               data/raw/tmdb_cast.parquet          (series_id, person_id, person_name, order, episode_count)
               data/raw/tmdb_creators.parquet      (series_id, person_id, person_name)
               data/raw/tmdb_directors.parquet     (series_id, person_id, person_name, episode_count)
               data/raw/tmdb_failed.parquet        (id) — для --retry

Всё на английском (language=en-US — у TMDB там лучшее покрытие). Короткие/
пустые overview (меньше ~4 предложений) дальше дополняет wiki_fallback.py
статьёй из Wikipedia (с переводом, если статья не на английском).

Схема Series/рёбер соответствует graph-series_backend (schema_init.cypher,
routers/*.py) — это то, что грузит neo4j_loader.py/qdrant_loader.py дальше.

Запуск:
  python tmdb_fetcher.py                    # discover + details, полный прогон
  python tmdb_fetcher.py --discover         # только discover
  python tmdb_fetcher.py --details          # только details (нужен tmdb_series_ids.parquet)
  python tmdb_fetcher.py --details --limit 5   # смоук-тест на 5 сериалах
  python tmdb_fetcher.py --details --retry     # повторить tmdb_failed.parquet
  python tmdb_fetcher.py --details --rps 20 --workers 5
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import logging
import os
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("tmdb_fetcher")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_API     = "https://api.themoviedb.org/3"
EXPORT_URL   = "https://files.tmdb.org/p/exports/tv_series_ids_{date}.json.gz"

LANGUAGE = "en-US"   # все данные — на английском (лучшее покрытие у TMDB); короткие/пустые
                     # overview дальше дополняет wiki_fallback.py (Wikipedia + перевод)
APPEND   = "external_ids,keywords,aggregate_credits,content_ratings"

RATE_LIMIT_RPS = 30.0   # текущий неофициальный потолок TMDB ~40 rps — берём с запасом
WORKERS        = 10
CHECKPOINT_EVERY = 10_000   # id между записями на диск (оборванный прогон не теряет сделанное)
TIMEOUT        = 30
MAX_RETRIES    = 5
BACKOFF_429    = [10, 30, 60, 120]

RAW_DIR      = Path("data/raw")
IDS_PATH     = RAW_DIR / "tmdb_series_ids.parquet"
SERIES_PATH  = RAW_DIR / "tmdb_series.parquet"
GENRES_PATH  = RAW_DIR / "tmdb_genres.parquet"
KEYWORDS_PATH   = RAW_DIR / "tmdb_keywords.parquet"
NETWORKS_PATH   = RAW_DIR / "tmdb_networks.parquet"
COUNTRIES_PATH  = RAW_DIR / "tmdb_countries.parquet"
LANGUAGES_PATH  = RAW_DIR / "tmdb_languages.parquet"
CAST_PATH       = RAW_DIR / "tmdb_cast.parquet"
CREATORS_PATH   = RAW_DIR / "tmdb_creators.parquet"
DIRECTORS_PATH  = RAW_DIR / "tmdb_directors.parquet"
FAILED_PATH     = RAW_DIR / "tmdb_failed.parquet"


def _require_api_key() -> None:
    if not TMDB_API_KEY:
        raise SystemExit(
            "TMDB_API_KEY не задан — добавь его в .env "
            "(ключ получают на https://www.themoviedb.org/settings/api)"
        )


# ---------------------------------------------------------------------------
# discover — daily ID export
# ---------------------------------------------------------------------------
def _download_export(days_back: int = 0) -> bytes:
    date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%m_%d_%Y")
    url = EXPORT_URL.format(date=date)
    resp = requests.get(url, timeout=60)
    if resp.status_code == 200:
        log.info("✔ Export найден: %s", url)
        return resp.content
    log.warning("  %s недоступен (%d)", url, resp.status_code)
    raise FileNotFoundError(url)


def run_discover() -> None:
    log.info("Скачиваем daily ID export (TV series)...")
    raw = None
    for days_back in range(3):   # файл публикуется к 8:00 UTC — за сегодня может ещё не быть
        try:
            raw = _download_export(days_back)
            break
        except FileNotFoundError:
            continue
    if raw is None:
        raise SystemExit("Не удалось скачать daily export ни за последние 3 дня")

    lines = gzip.decompress(raw).decode("utf-8").strip().split("\n")
    log.info("Строк в export: %d", len(lines))

    rows = []
    for line in lines:
        if not line:
            continue
        obj = pd.io.json.ujson_loads(line) if hasattr(pd.io.json, "ujson_loads") else __import__("json").loads(line)
        if obj.get("adult"):
            continue
        rows.append({
            "tmdb_id":       obj["id"],
            "name":          obj.get("original_name") or obj.get("name") or "",
            "popularity":    obj.get("popularity", 0.0),
        })

    df = pd.DataFrame(rows).drop_duplicates(subset="tmdb_id")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(IDS_PATH, index=False, compression="zstd")
    log.info("✔ Сохранено %d id → %s", len(df), IDS_PATH)


# ---------------------------------------------------------------------------
# details — комбинированный запрос на серию
# ---------------------------------------------------------------------------
class RateLimiter:
    """Токен-бакет: не более rps запросов в секунду глобально."""

    def __init__(self, rps: float) -> None:
        self._interval = 1.0 / rps
        self._lock = asyncio.Lock()
        self._last_call: float | None = None

    async def acquire(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            if self._last_call is not None:
                wait = self._interval - (now - self._last_call)
                if wait > 0:
                    await asyncio.sleep(wait)
            self._last_call = asyncio.get_event_loop().time()


def _year(date_str: str | None) -> int | None:
    if not date_str:
        return None
    try:
        return int(date_str[:4])
    except (ValueError, TypeError):
        return None


def _us_rating(content_ratings: dict | None) -> str:
    for entry in (content_ratings or {}).get("results", []):
        if entry.get("iso_3166_1") == "US":
            return entry.get("rating") or ""
    return ""


def parse_series(data: dict) -> dict:
    external = data.get("external_ids") or {}
    return {
        "tmdb_id":           data["id"],
        "name":              data.get("name") or "",
        "original_name":     data.get("original_name") or "",
        "imdb_id":           external.get("imdb_id") or "",
        "wikidata_id":       external.get("wikidata_id") or "",
        "tvdb_id":           external.get("tvdb_id"),
        "start_year":        _year(data.get("first_air_date")),
        "end_year":          _year(data.get("last_air_date")),
        "status":            data.get("status") or "",
        "type":              data.get("type") or "",
        "in_production":     bool(data.get("in_production", False)),
        "original_language": data.get("original_language") or "",
        "season_count":      data.get("number_of_seasons") or 0,
        "episode_count":     data.get("number_of_episodes") or 0,
        "episode_runtime":   (data.get("episode_run_time") or [None])[0],
        "popularity":        data.get("popularity") or 0.0,
        "vote_average":      data.get("vote_average") or 0.0,
        "vote_count":        data.get("vote_count") or 0,
        "us_content_rating": _us_rating(data.get("content_ratings")),
        "overview":          data.get("overview") or "",
        "overview_source":   "tmdb",   # wiki_fallback.py переставит на "wikipedia" для дополненных строк
        "tagline":           data.get("tagline") or "",
        "homepage":          data.get("homepage") or "",
        "poster_path":       data.get("poster_path") or "",
    }


def parse_relations(data: dict) -> dict[str, list[dict]]:
    sid = data["id"]

    genres = [
        {"series_id": sid, "genre_id": g["id"], "genre_name": g["name"]}
        for g in data.get("genres", [])
    ]
    keywords = [
        {"series_id": sid, "keyword_id": k["id"], "keyword_name": k["name"]}
        for k in (data.get("keywords") or {}).get("results", [])
    ]
    networks = [
        {"series_id": sid, "network_id": n["id"], "network_name": n.get("name") or "",
         "country": n.get("origin_country") or ""}
        for n in data.get("networks", [])
    ]
    countries = [
        {"series_id": sid, "code": c["iso_3166_1"]}
        for c in data.get("production_countries", [])
    ]
    languages = [
        {"series_id": sid, "code": code}
        for code in (data.get("languages") or [])
    ]
    creators = [
        {"series_id": sid, "person_id": p["id"], "person_name": p.get("name") or ""}
        for p in data.get("created_by", [])
    ]

    credits = data.get("aggregate_credits") or {}
    cast = []
    for actor in credits.get("cast", []):
        roles = actor.get("roles") or []
        episode_count = sum(r.get("episode_count", 0) for r in roles) or actor.get("total_episode_count", 0)
        cast.append({
            "series_id": sid, "person_id": actor["id"], "person_name": actor.get("name") or "",
            "order": actor.get("order"), "episode_count": episode_count,
        })
    directors = []
    for member in credits.get("crew", []):
        for job in member.get("jobs") or []:
            if job.get("job") == "Director":
                directors.append({
                    "series_id": sid, "person_id": member["id"], "person_name": member.get("name") or "",
                    "episode_count": job.get("episode_count", 0),
                })

    return {
        "genres": genres, "keywords": keywords, "networks": networks,
        "countries": countries, "languages": languages,
        "creators": creators, "cast": cast, "directors": directors,
    }


async def fetch_one(session: aiohttp.ClientSession, rate: RateLimiter, tmdb_id: int) -> dict | None:
    params = {
        "api_key": TMDB_API_KEY,
        "language": LANGUAGE,
        "append_to_response": APPEND,
    }
    for attempt in range(MAX_RETRIES):
        await rate.acquire()
        try:
            async with session.get(
                f"{TMDB_API}/tv/{tmdb_id}", params=params,
                timeout=aiohttp.ClientTimeout(total=TIMEOUT),
            ) as resp:
                if resp.status == 404:
                    return None   # сериал удалён из TMDB с момента export — не ошибка, просто пропускаем
                if resp.status == 429:
                    wait = BACKOFF_429[min(attempt, len(BACKOFF_429) - 1)]
                    log.warning("429 (id=%d, попытка %d/%d) — спим %d сек", tmdb_id, attempt + 1, MAX_RETRIES, wait)
                    await asyncio.sleep(wait)
                    continue
                if resp.status >= 500:
                    await asyncio.sleep(5.0)
                    continue
                resp.raise_for_status()
                return await resp.json()
        except asyncio.TimeoutError:
            await asyncio.sleep(3.0)
        except aiohttp.ClientError as exc:
            log.debug("ClientError id=%d: %s", tmdb_id, exc)
            await asyncio.sleep(3.0)
    return "error"   # исчерпали retries — отличаем от 404 (None)


async def worker(
    queue: asyncio.Queue, session: aiohttp.ClientSession, rate: RateLimiter,
    series_rows: list, relation_rows: dict[str, list], failed_ids: list, stats: Counter,
) -> None:
    while True:
        tmdb_id = await queue.get()
        if tmdb_id is None:
            queue.task_done()
            break

        data = await fetch_one(session, rate, tmdb_id)
        if data is None:
            stats["not_found"] += 1
        elif data == "error":
            failed_ids.append(tmdb_id)
            stats["error"] += 1
        else:
            series_rows.append(parse_series(data))
            for key, rows in parse_relations(data).items():
                relation_rows[key].extend(rows)
            stats["ok"] += 1

        queue.task_done()


async def fetch_all(ids: list[int], workers: int, rps: float) -> tuple[list, dict[str, list], list, Counter]:
    series_rows: list = []
    relation_rows: dict[str, list] = {
        k: [] for k in ("genres", "keywords", "networks", "countries", "languages", "creators", "cast", "directors")
    }
    failed_ids: list = []
    stats: Counter = Counter()
    rate = RateLimiter(rps)

    queue: asyncio.Queue = asyncio.Queue()
    for tmdb_id in ids:
        queue.put_nowait(tmdb_id)
    for _ in range(workers):
        queue.put_nowait(None)

    total = len(ids)
    t0 = time.time()

    async def progress_logger() -> None:
        while True:
            await asyncio.sleep(30)
            done = sum(stats.values())
            if done == 0:
                continue
            elapsed = time.time() - t0
            rps_cur = done / elapsed
            eta = (total - done) / rps_cur if rps_cur > 0 else 0
            log.info(
                "  Прогресс: %d / %d  |  %.1f req/s  |  ETA: %.1f мин  |  ok: %d  not_found: %d  error: %d",
                done, total, rps_cur, eta / 60, stats["ok"], stats["not_found"], stats["error"],
            )
            if done >= total:
                break

    connector = aiohttp.TCPConnector(limit=workers + 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        worker_tasks = [
            asyncio.create_task(worker(queue, session, rate, series_rows, relation_rows, failed_ids, stats))
            for _ in range(workers)
        ]
        progress_task = asyncio.create_task(progress_logger())
        await asyncio.gather(*worker_tasks)
        progress_task.cancel()

    elapsed = time.time() - t0
    done = sum(stats.values())
    log.info(
        "  Финал: %d / %d  |  %.1f req/s  |  ok: %d  not_found: %d  error: %d",
        done, total, done / elapsed if elapsed > 0 else 0, stats["ok"], stats["not_found"], stats["error"],
    )
    return series_rows, relation_rows, failed_ids, stats


# ---------------------------------------------------------------------------
# Сохранение — merge с существующими файлами (resume idiom)
# ---------------------------------------------------------------------------
def _merge_save(rows: list[dict], path: Path, dedup_subset: list[str] | None = None) -> None:
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, new_df], ignore_index=True)
        if dedup_subset:
            df = df.drop_duplicates(subset=dedup_subset, keep="last")
    else:
        df = new_df
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")


def save_results(series_rows: list, relation_rows: dict[str, list]) -> None:
    # Связи пишем раньше series и с дедупликацией по ключу: серию считают «загруженной» по
    # tmdb_series.parquet, так что если прогон оборвётся между файлами, её перекачают, а дубли
    # связей не накопятся.
    _merge_save(relation_rows["genres"], GENRES_PATH, ["series_id", "genre_id"])
    _merge_save(relation_rows["keywords"], KEYWORDS_PATH, ["series_id", "keyword_id"])
    _merge_save(relation_rows["networks"], NETWORKS_PATH, ["series_id", "network_id"])
    _merge_save(relation_rows["countries"], COUNTRIES_PATH, ["series_id", "code"])
    _merge_save(relation_rows["languages"], LANGUAGES_PATH, ["series_id", "code"])
    _merge_save(relation_rows["cast"], CAST_PATH, ["series_id", "person_id"])
    _merge_save(relation_rows["creators"], CREATORS_PATH, ["series_id", "person_id"])
    _merge_save(relation_rows["directors"], DIRECTORS_PATH, ["series_id", "person_id"])
    _merge_save(series_rows, SERIES_PATH, dedup_subset=["tmdb_id"])

    if series_rows:
        log.info("✔ Series: +%d → %s", len(series_rows), SERIES_PATH)


def save_failed(failed_ids: list) -> None:
    if failed_ids:
        pd.DataFrame({"tmdb_id": failed_ids}).to_parquet(FAILED_PATH, index=False, compression="zstd")
        log.info("⚠ %d с ошибками → %s  (--retry чтобы повторить)", len(failed_ids), FAILED_PATH)
    elif FAILED_PATH.exists():
        FAILED_PATH.unlink()
        log.info("✔ Ошибок нет, tmdb_failed.parquet удалён")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------
def run_details(resume: bool = True, retry: bool = False, limit: int | None = None,
                 workers: int = WORKERS, rps: float = RATE_LIMIT_RPS) -> None:
    _require_api_key()

    if retry:
        if not FAILED_PATH.exists():
            log.error("Файл %s не найден — нечего повторять", FAILED_PATH)
            return
        ids = pd.read_parquet(FAILED_PATH)["tmdb_id"].tolist()
        log.info("Режим retry: %d id из %s", len(ids), FAILED_PATH)
    else:
        if not IDS_PATH.exists():
            raise SystemExit(f"Не найден {IDS_PATH} — сначала прогони --discover")
        ids = pd.read_parquet(IDS_PATH)["tmdb_id"].tolist()
        if resume and SERIES_PATH.exists():
            done = set(pd.read_parquet(SERIES_PATH)["tmdb_id"].tolist())
            before = len(ids)
            ids = [i for i in ids if i not in done]
            log.info("Уже загружено: %d  |  Осталось: %d (из %d)", before - len(ids), len(ids), before)

    if limit:
        ids = ids[:limit]

    log.info("Серий к загрузке: %d  |  workers=%d  rps=%.0f", len(ids), workers, rps)
    if not ids:
        log.info("Нечего загружать.")
        return

    # Качаем и сохраняем порциями: оборванный прогон теряет максимум одну порцию.
    failed_all: list = []
    for start in range(0, len(ids), CHECKPOINT_EVERY):
        chunk = ids[start:start + CHECKPOINT_EVERY]
        log.info("Порция %d–%d из %d", start + 1, start + len(chunk), len(ids))
        series_rows, relation_rows, failed_ids, _ = asyncio.run(fetch_all(chunk, workers, rps))
        save_results(series_rows, relation_rows)
        failed_all.extend(failed_ids)
    save_failed(failed_all)


def main() -> None:
    parser = argparse.ArgumentParser(description="TMDB TV series fetcher")
    parser.add_argument("--discover", action="store_true", help="Только discover (daily ID export)")
    parser.add_argument("--details", action="store_true", help="Только details (по tmdb_series_ids.parquet)")
    parser.add_argument("--retry", action="store_true", help="Повторить только tmdb_failed.parquet")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число id (смоук-тест)")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--rps", type=float, default=RATE_LIMIT_RPS)
    parser.add_argument("--no-resume", action="store_true", help="Не пропускать уже загруженные id")
    args = parser.parse_args()

    do_discover = args.discover or not (args.discover or args.details)
    do_details  = args.details or not (args.discover or args.details)

    if do_discover:
        run_discover()
    if do_details:
        run_details(
            resume=not args.no_resume, retry=args.retry, limit=args.limit,
            workers=args.workers, rps=args.rps,
        )


if __name__ == "__main__":
    main()

"""
wiki_fallback.py
~~~~~~~~~~~~~~~~
Fallback-обогащение overview через Wikipedia для сериалов, у которых
TMDB-overview короче MIN_OVERVIEW_CHARS символов (или отсутствует) — для
не самых популярных сериалов на TMDB часто просто нет развёрнутого
описания (меньше ~4 предложений).

Источник — статья Wikipedia по sitelink из Wikidata (wikidata_id, который
TMDB отдаёт в external_ids). Предпочитаем enwiki; если его нет — статью
на original_language сериала; если и её нет — любую статью на языке из
NLLB_LANG. Не-английские статьи переводим локальной моделью NLLB-200
(без API-ключей, оффлайн — модель качается один раз и кэшируется).

Вход/выход: data/raw/tmdb_series.parquet — обновляет overview и
overview_source ("tmdb" / "wikipedia") только у дополненных строк.

Запуск:
  python wiki_fallback.py
  python wiki_fallback.py --limit 100        # смоук-тест
  python wiki_fallback.py --min-chars 300    # другой порог
  python wiki_fallback.py --device cpu       # без GPU (медленнее)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
import urllib.parse
from pathlib import Path

import aiohttp
import pandas as pd

log = logging.getLogger("wiki_fallback")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------
MIN_OVERVIEW_CHARS = 200

WIKIDATA_API   = "https://www.wikidata.org/w/api.php"
WIKIDATA_BATCH = 50    # id за один wbgetentities

RATE_LIMIT_RPS = 10.0
WORKERS        = 8
TIMEOUT        = 20
MAX_RETRIES    = 3

TRANSLATE_MODEL = "facebook/nllb-200-distilled-600M"
TRANSLATE_BATCH = 16
CHECKPOINT_EVERY = 2000   # кандидатов между записями parquet

HEADERS = {"User-Agent": "TVSeriesKnowledgeGraph/1.0 (graph-series_ETL; contact: github.com/GKatzer)"}

SERIES_PATH = Path("data/raw/tmdb_series.parquet")

# Wikipedia site-код → NLLB-200 (FLORES-200) код. Языки вне списка — не
# переводим (пропускаем как источник), их слишком мало среди сериальных
# статей, чтобы расширять список дальше.
NLLB_LANG: dict[str, str] = {
    "ja": "jpn_Jpan", "fr": "fra_Latn", "de": "deu_Latn", "ru": "rus_Cyrl",
    "es": "spa_Latn", "it": "ita_Latn", "pt": "por_Latn", "zh": "zho_Hans",
    "ko": "kor_Hang", "ar": "arb_Arab", "hi": "hin_Deva", "tr": "tur_Latn",
    "pl": "pol_Latn", "nl": "nld_Latn", "sv": "swe_Latn", "fi": "fin_Latn",
    "da": "dan_Latn", "no": "nob_Latn", "cs": "ces_Latn", "el": "ell_Grek",
    "he": "heb_Hebr", "th": "tha_Thai", "vi": "vie_Latn", "id": "ind_Latn",
    "uk": "ukr_Cyrl", "ro": "ron_Latn", "hu": "hun_Latn", "fa": "pes_Arab",
    "sr": "srp_Cyrl", "hr": "hrv_Latn", "bg": "bul_Cyrl",
}

# Служебные sitelink-проекты (не статьи) — пропускаем при переборе языков
NON_ARTICLE_SUFFIXES = ("wikidatawiki", "commonswiki", "specieswiki", "metawiki",
                         "mediawikiwiki", "wikifunctionswiki", "wikimaniawiki")


# ---------------------------------------------------------------------------
# Rate limiter — тот же токен-бакет, что в tmdb_fetcher.py / wiki_texts.py
# ---------------------------------------------------------------------------
class RateLimiter:
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


# ---------------------------------------------------------------------------
# Wikidata sitelinks → выбор источника
# ---------------------------------------------------------------------------
async def get_sitelinks_batch(session: aiohttp.ClientSession, wikidata_ids: list[str]) -> dict[str, dict[str, str]]:
    """{wikidata_id: {site_lang: title}} — только языковые статьи (без commons и т.п.)."""
    params = {
        "action": "wbgetentities",
        "ids": "|".join(wikidata_ids),
        "props": "sitelinks",
        "format": "json",
    }
    async with session.get(WIKIDATA_API, params=params, timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as resp:
        data = await resp.json()

    out: dict[str, dict[str, str]] = {}
    for qid, entity in (data.get("entities") or {}).items():
        links = entity.get("sitelinks") or {}
        langs: dict[str, str] = {}
        for site, info in links.items():
            if not site.endswith("wiki") or site in NON_ARTICLE_SUFFIXES:
                continue
            lang = site[: -len("wiki")]
            langs[lang] = info["title"]
        out[qid] = langs
    return out


def pick_source(sitelinks: dict[str, str], original_language: str) -> tuple[str, str] | None:
    """Возвращает (lang, title) — приоритет: enwiki → original_language → первый переводимый."""
    if "en" in sitelinks:
        return "en", sitelinks["en"]
    if original_language and original_language in sitelinks:
        return original_language, sitelinks[original_language]
    for lang, title in sitelinks.items():
        if lang in NLLB_LANG:
            return lang, title
    return None


# ---------------------------------------------------------------------------
# Wikipedia summary (REST, одна статья за раз — как retry-режим в старом wiki_texts.py)
# ---------------------------------------------------------------------------
async def fetch_summary(session: aiohttp.ClientSession, rate: RateLimiter, lang: str, title: str) -> str | None:
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title, safe='')}"
    for attempt in range(MAX_RETRIES):
        await rate.acquire()
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as resp:
                if resp.status == 404:
                    return None
                if resp.status == 429:
                    await asyncio.sleep(10)
                    continue
                if resp.status >= 500:
                    await asyncio.sleep(5)
                    continue
                data = await resp.json()
                extract = (data.get("extract") or "").strip()
                return extract or None
        except (asyncio.TimeoutError, aiohttp.ClientError):
            await asyncio.sleep(3)
    return None


# ---------------------------------------------------------------------------
# Асинхронный сбор кандидатов
# ---------------------------------------------------------------------------
async def collect_fallbacks(candidates: pd.DataFrame, workers: int, rps: float) -> dict[int, tuple[str, str]]:
    """Возвращает {row_index: (lang, raw_text)} для успешно найденных статей."""
    rate = RateLimiter(rps)
    results: dict[int, tuple[str, str]] = {}

    connector = aiohttp.TCPConnector(limit=workers + 2)
    async with aiohttp.ClientSession(headers=HEADERS, connector=connector) as session:
        # 1. sitelinks батчами по WIKIDATA_BATCH
        rows = list(candidates.itertuples())
        sitelinks_map: dict[str, dict[str, str]] = {}
        for i in range(0, len(rows), WIKIDATA_BATCH):
            chunk = rows[i : i + WIKIDATA_BATCH]
            ids = [r.wikidata_id for r in chunk]
            await rate.acquire()
            sitelinks_map.update(await get_sitelinks_batch(session, ids))

        # 2. выбор источника + очередь на fetch_summary
        queue: asyncio.Queue = asyncio.Queue()
        for r in rows:
            sitelinks = sitelinks_map.get(r.wikidata_id, {})
            picked = pick_source(sitelinks, r.original_language)
            if picked:
                queue.put_nowait((r.Index, picked[0], picked[1]))
        total = queue.qsize()
        for _ in range(workers):
            queue.put_nowait(None)

        log.info("Найдено кандидатов со статьёй Wikipedia: %d / %d", total, len(rows))

        async def worker() -> None:
            while True:
                item = await queue.get()
                if item is None:
                    queue.task_done()
                    break
                idx, lang, title = item
                text = await fetch_summary(session, rate, lang, title)
                if text:
                    results[idx] = (lang, text)
                queue.task_done()

        await asyncio.gather(*(worker() for _ in range(workers)))

    return results


# ---------------------------------------------------------------------------
# Перевод — батчами по языку-источнику, модель грузится один раз.
# NLLB грузим напрямую через AutoModelForSeq2SeqLM/AutoTokenizer, а не через
# pipeline("translation", ...) — в новых версиях transformers этой задачи
# в реестре pipeline() больше нет (см. PIPELINE_REGISTRY.get_supported_tasks()).
# ---------------------------------------------------------------------------
_translate_model = None
_translate_tokenizer = None


def _get_translate_model(device: str):
    global _translate_model, _translate_tokenizer
    if _translate_model is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        log.info("Загружаем модель перевода %s (device=%s)...", TRANSLATE_MODEL, device)
        _translate_tokenizer = AutoTokenizer.from_pretrained(TRANSLATE_MODEL)
        _translate_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATE_MODEL).to(device)
        _translate_model.eval()
    return _translate_model, _translate_tokenizer


def translate_grouped(by_lang: dict[str, list[tuple[int, str]]], device: str) -> dict[int, str]:
    """by_lang: {lang: [(row_index, text), ...]} → {row_index: english_text}."""
    out: dict[int, str] = {}

    non_english = {lang: items for lang, items in by_lang.items() if lang != "en"}
    out.update({idx: text for idx, text in by_lang.get("en", [])})
    if not non_english:
        return out

    import torch
    model, tokenizer = _get_translate_model(device)
    eng_bos = tokenizer.convert_tokens_to_ids("eng_Latn")

    for lang, items in non_english.items():
        nllb_src = NLLB_LANG[lang]
        log.info("Переводим %d текстов с '%s' (%s)...", len(items), lang, nllb_src)
        tokenizer.src_lang = nllb_src
        texts = [text for _, text in items]
        for i in range(0, len(texts), TRANSLATE_BATCH):
            batch_items = items[i : i + TRANSLATE_BATCH]
            batch_texts = texts[i : i + TRANSLATE_BATCH]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                                truncation=True, max_length=512).to(device)
            with torch.no_grad():
                generated = model.generate(**inputs, forced_bos_token_id=eng_bos, max_length=1024)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for (idx, _), text in zip(batch_items, decoded):
                out[idx] = text
    return out


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------
def run(limit: int | None = None, min_chars: int = MIN_OVERVIEW_CHARS,
        workers: int = WORKERS, rps: float = RATE_LIMIT_RPS, device: str = "cuda") -> None:
    log.info("Загружаем %s...", SERIES_PATH)
    df = pd.read_parquet(SERIES_PATH)
    if "overview_source" not in df.columns:
        df["overview_source"] = "tmdb"

    overview_len = df["overview"].fillna("").str.strip().str.len()
    candidates_mask = (
        (df["overview_source"] != "wikipedia")
        & (overview_len < min_chars)
        & (df["wikidata_id"].fillna("").str.strip().astype(bool))
    )
    candidates = df[candidates_mask]
    if limit:
        candidates = candidates.head(limit)
    log.info("Кандидатов на Wikipedia-fallback (overview < %d символов): %d", min_chars, len(candidates))

    if candidates.empty:
        log.info("Нечего обогащать.")
        return

    # Порциями: скачали → перевели → записали parquet. Оборванный прогон теряет максимум
    # одну порцию, перезапуск берёт только строки, где overview_source ещё не "wikipedia".
    for start in range(0, len(candidates), CHECKPOINT_EVERY):
        chunk = candidates.iloc[start:start + CHECKPOINT_EVERY]
        log.info("Порция %d–%d из %d", start + 1, start + len(chunk), len(candidates))

        t0 = time.time()
        fetched = asyncio.run(collect_fallbacks(chunk, workers, rps))
        log.info("✔ Статей найдено и загружено: %d за %.1f сек", len(fetched), time.time() - t0)
        if not fetched:
            continue

        by_lang: dict[str, list[tuple[int, str]]] = {}
        for idx, (lang, text) in fetched.items():
            by_lang.setdefault(lang, []).append((idx, text))

        t0 = time.time()
        english_texts = translate_grouped(by_lang, device)
        log.info("✔ Перевод завершён за %.1f сек", time.time() - t0)

        for idx, text in english_texts.items():
            df.loc[idx, "overview"] = text
            df.loc[idx, "overview_source"] = "wikipedia"

        df.to_parquet(SERIES_PATH, index=False, compression="zstd")
        log.info("✔ Обновлено %d строк → %s", len(english_texts), SERIES_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wikipedia fallback для коротких TMDB-overview")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число кандидатов (смоук-тест)")
    parser.add_argument("--min-chars", type=int, default=MIN_OVERVIEW_CHARS,
                         help=f"Порог длины overview (default {MIN_OVERVIEW_CHARS})")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--rps", type=float, default=RATE_LIMIT_RPS)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    run(limit=args.limit, min_chars=args.min_chars, workers=args.workers, rps=args.rps, device=args.device)


if __name__ == "__main__":
    main()

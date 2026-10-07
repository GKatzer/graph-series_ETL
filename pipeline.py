"""
pipeline.py — TV Series ETL Pipeline (TMDB)

Шаги:
  1. tmdb_discover — daily ID export (все известные tmdb_id)  → data/raw/tmdb_series_ids.parquet
  2. tmdb_details  — детали по каждому id (1 запрос/сериал)   → data/raw/tmdb_*.parquet
  3. wiki_fallback — Wikipedia+перевод для коротких overview   → обновляет data/raw/tmdb_series.parquet
  4. embeddings    — BGE-эмбеддинги name+overview+keywords     → data/processed/tmdb_embeddings.parquet
  5. qdrant_loader — upsert в Qdrant (point id = tmdb_id)
  6. neo4j_loader  — загрузка графа (vote_count >= 2)

Запуск:
  python pipeline.py                                          # все шаги
  python pipeline.py --steps tmdb_discover tmdb_details        # первые два
  python pipeline.py --resume                                  # пропустить готовый discover (остальные докачивают сами)
  python pipeline.py --device cpu                              # без GPU
"""
from __future__ import annotations
import argparse, logging, sys, time
from pathlib import Path

log = logging.getLogger("pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

STEP_OUTPUTS: dict[str, list[Path]] = {
    "tmdb_discover": [Path("data/raw/tmdb_series_ids.parquet")],
    # Шаги ниже докачивают с места остановки сами (details — по id, embeddings — по tmdb_id,
    # wiki_fallback — по overview_source), поэтому наличие файла не значит «готово»:
    # --resume их не пропускает, а просто запускает — готовое они отсекут сами.
    "tmdb_details":  [],
    "wiki_fallback": [],
    "embeddings":    [],
    "qdrant_loader": [],
    "neo4j_loader":  [],
}
ALL_STEPS = list(STEP_OUTPUTS.keys())

def step_tmdb_discover() -> None:
    from tmdb_fetcher import run_discover
    log.info("═══ ШАГ 1: TMDB daily ID export ═══")
    run_discover()

def step_tmdb_details(limit: int | None = None) -> None:
    from tmdb_fetcher import run_details
    log.info("═══ ШАГ 2: TMDB детали по сериалам ═══")
    run_details(limit=limit)

def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

def step_wiki_fallback(limit: int | None = None, device: str = "cpu") -> None:
    from wiki_fallback import run as wiki_fallback_run
    log.info("═══ ШАГ 3: Wikipedia fallback для коротких overview ═══")
    wiki_fallback_run(limit=limit, device=device)

def step_embeddings(device: str = "cpu") -> None:
    from embeddings import run as embeddings_run
    log.info("═══ ШАГ 4: BGE-эмбеддинги ═══")
    embeddings_run(device=device)

def step_qdrant_loader() -> None:
    from qdrant_loader import run as qdrant_run
    log.info("═══ ШАГ 5: Загрузка в Qdrant ═══")
    qdrant_run()

def step_neo4j_loader() -> None:
    from neo4j_loader import run as neo4j_run
    log.info("═══ ШАГ 6: Загрузка в Neo4j ═══")
    neo4j_run()

def _is_done(step: str) -> bool:
    outputs = STEP_OUTPUTS.get(step, [])
    if not outputs:
        return False   # шаги без файлового выхода никогда не считаются "готовыми"
    return all(p.exists() and p.stat().st_size > 0 for p in outputs)

def run(steps: list[str], resume: bool = False, limit: int | None = None, device: str = "auto") -> None:
    t0 = time.time()
    errors: list[str] = []
    if {"wiki_fallback", "embeddings"} & set(steps):
        device = resolve_device(device)
        log.info("Устройство для перевода/эмбеддингов: %s", device)
    for step in steps:
        if resume and _is_done(step):
            log.info("⏭  Пропускаем '%s' (output уже есть)", step)
            continue
        t_step = time.time()
        try:
            if step == "tmdb_discover":   step_tmdb_discover()
            elif step == "tmdb_details":  step_tmdb_details(limit=limit)
            elif step == "wiki_fallback": step_wiki_fallback(limit=limit, device=device)
            elif step == "embeddings":    step_embeddings(device=device)
            elif step == "qdrant_loader": step_qdrant_loader()
            elif step == "neo4j_loader":  step_neo4j_loader()
            log.info("✔  '%s' завершён за %.1f сек", step, time.time() - t_step)
        except Exception as exc:
            log.error("✗  Шаг '%s' упал: %s", step, exc)
            errors.append(step)
    log.info("Пайплайн завершён за %.1f сек. Ошибки: %s", time.time() - t0, errors or "нет")
    if errors:
        sys.exit(1)

def main() -> None:
    parser = argparse.ArgumentParser(description="TV Series ETL Pipeline (TMDB)")
    parser.add_argument("--steps", nargs="+", choices=ALL_STEPS, default=ALL_STEPS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                         help="Ограничить tmdb_details/wiki_fallback числом id (смоук-тест)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                         help="Устройство для перевода и эмбеддингов (auto: cuda, если доступна)")
    args = parser.parse_args()
    run(steps=args.steps, resume=args.resume, limit=args.limit, device=args.device)

if __name__ == "__main__":
    main()

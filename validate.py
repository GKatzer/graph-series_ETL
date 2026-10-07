"""
validate.py
~~~~~~~~~~~
Проверяет все Parquet-файлы после ETL:
- файл существует и не пустой
- ключевые колонки присутствуют
- нет полностью пустых строк
- базовая статистика (строк, уникальных ID, % пустых значений)

Запуск:
  python validate.py           # полный отчёт
  python validate.py --fix     # показать проблемные строки подробнее
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ожидаемые файлы и их схемы
# ---------------------------------------------------------------------------
@dataclass
class FileSpec:
    path: Path
    required_columns: list[str]
    id_column: str                    # колонка для подсчёта уникальных значений
    min_rows: int = 1000              # минимум строк — если меньше, это подозрительно


SPECS: list[FileSpec] = [
    # --- RAW ---
    FileSpec(
        path=Path("data/raw/tmdb_series_ids.parquet"),
        required_columns=["tmdb_id", "name"],
        id_column="tmdb_id",
        min_rows=150_000,
    ),
    FileSpec(
        path=Path("data/raw/tmdb_series.parquet"),
        required_columns=["tmdb_id", "name", "overview", "overview_source", "vote_count"],
        id_column="tmdb_id",
        min_rows=150_000,
    ),
    FileSpec(
        path=Path("data/raw/tmdb_genres.parquet"),
        required_columns=["series_id", "genre_id", "genre_name"],
        id_column="series_id",
        min_rows=100_000,
    ),
    FileSpec(
        path=Path("data/raw/tmdb_keywords.parquet"),
        required_columns=["series_id", "keyword_id", "keyword_name"],
        id_column="series_id",
        min_rows=100_000,
    ),
    FileSpec(
        path=Path("data/raw/tmdb_networks.parquet"),
        required_columns=["series_id", "network_id", "network_name"],
        id_column="series_id",
        min_rows=50_000,
    ),
    FileSpec(
        path=Path("data/raw/tmdb_countries.parquet"),
        required_columns=["series_id", "code"],
        id_column="series_id",
        min_rows=50_000,
    ),
    FileSpec(
        path=Path("data/raw/tmdb_languages.parquet"),
        required_columns=["series_id", "code"],
        id_column="series_id",
        min_rows=50_000,
    ),
    FileSpec(
        path=Path("data/raw/tmdb_cast.parquet"),
        required_columns=["series_id", "person_id", "person_name"],
        id_column="series_id",
        min_rows=100_000,
    ),
    FileSpec(
        path=Path("data/raw/tmdb_creators.parquet"),
        required_columns=["series_id", "person_id", "person_name"],
        id_column="series_id",
        min_rows=10_000,
    ),
    FileSpec(
        path=Path("data/raw/tmdb_directors.parquet"),
        required_columns=["series_id", "person_id", "person_name"],
        id_column="series_id",
        min_rows=10_000,
    ),
    # --- PROCESSED ---
    FileSpec(
        path=Path("data/processed/tmdb_embeddings.parquet"),
        required_columns=["tmdb_id", "embedding"],
        id_column="tmdb_id",
        min_rows=150_000,
    ),
]

# ---------------------------------------------------------------------------
# Результат проверки одного файла
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    path: Path
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Проверка одного файла
# ---------------------------------------------------------------------------
def check_file(spec: FileSpec, verbose: bool = False) -> CheckResult:
    result = CheckResult(path=spec.path)

    # 1. Существование
    if not spec.path.exists():
        result.ok = False
        result.errors.append("файл не найден")
        return result

    size_mb = spec.path.stat().st_size / 1e6
    if size_mb == 0:
        result.ok = False
        result.errors.append("файл пустой (0 байт)")
        return result

    # 2. Читаемость
    try:
        df = pd.read_parquet(spec.path)
    except Exception as exc:
        result.ok = False
        result.errors.append(f"не удалось прочитать: {exc}")
        return result

    result.stats["rows"] = len(df)
    result.stats["size_mb"] = round(size_mb, 1)
    result.stats["columns"] = list(df.columns)

    # 3. Колонки
    missing_cols = [c for c in spec.required_columns if c not in df.columns]
    if missing_cols:
        result.ok = False
        result.errors.append(f"отсутствуют колонки: {missing_cols}")

    # 4. Минимум строк
    if len(df) < spec.min_rows:
        result.ok = False
        result.errors.append(
            f"строк {len(df):,} — меньше ожидаемого минимума {spec.min_rows:,}"
        )

    # 5. Уникальные ID
    if spec.id_column in df.columns:
        unique_ids = df[spec.id_column].nunique()
        result.stats["unique_ids"] = unique_ids

    # 6. Пустые значения по колонкам
    null_stats: dict[str, str] = {}
    for col in spec.required_columns:
        if col not in df.columns:
            continue
        null_pct = df[col].isna().mean() * 100
        null_stats[col] = f"{null_pct:.1f}%"
        if null_pct > 20:
            result.warnings.append(f"колонка '{col}' пустая на {null_pct:.1f}%")
    result.stats["null_pct"] = null_stats

    # 7. Полностью дублирующиеся строки (пропускаем колонки с нехэшируемыми
    # значениями, напр. embedding: list[float]/ndarray — там дубли смысла не имеют)
    def _is_hashable_col(col: pd.Series) -> bool:
        sample = col.dropna()
        return sample.empty or not isinstance(sample.iloc[0], (list, dict, set, np.ndarray))

    hashable_cols = [c for c in df.columns if _is_hashable_col(df[c])]
    if hashable_cols:
        dup_count = df[hashable_cols].duplicated().sum()
        if dup_count > 0:
            dup_pct = dup_count / len(df) * 100
            result.stats["duplicates"] = dup_count
            if dup_pct > 5:
                result.warnings.append(f"дублей {dup_count:,} ({dup_pct:.1f}%)")

    # 8. Выборка проблемных строк (verbose)
    if verbose and spec.id_column in df.columns:
        null_rows = df[df[spec.id_column].isna()]
        if not null_rows.empty:
            result.stats["null_id_sample"] = null_rows.head(3).to_dict("records")

    return result


# ---------------------------------------------------------------------------
# Печать отчёта
# ---------------------------------------------------------------------------
RESET  = "\033[0m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"


def _color(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


def print_report(results: list[CheckResult]) -> bool:
    total = len(results)
    passed = sum(1 for r in results if r.ok and not r.warnings)
    warned = sum(1 for r in results if r.ok and r.warnings)
    failed = sum(1 for r in results if not r.ok)

    print()
    print(_color("═" * 60, BOLD))
    print(_color("  ETL Validation Report", BOLD))
    print(_color("═" * 60, BOLD))

    for r in results:
        # Статус
        if not r.ok:
            status = _color("✗ FAIL", RED)
        elif r.warnings:
            status = _color("⚠ WARN", YELLOW)
        else:
            status = _color("✔ OK  ", GREEN)

        print(f"\n{status}  {_color(str(r.path), CYAN)}")

        # Статистика
        s = r.stats
        if "rows" in s:
            uid = f"  уникальных ID: {s['unique_ids']:,}" if "unique_ids" in s else ""
            print(f"       строк: {s['rows']:,}{uid}  |  размер: {s['size_mb']} MB")

        if "null_pct" in s and s["null_pct"]:
            nulls = "  |  ".join(f"{k}: {v}" for k, v in s["null_pct"].items())
            print(f"       null%: {nulls}")

        if "duplicates" in s:
            print(f"       дублей: {s['duplicates']:,}")

        # Ошибки
        for err in r.errors:
            print(f"       {_color('ERROR', RED)}: {err}")

        # Предупреждения
        for warn in r.warnings:
            print(f"       {_color('WARN', YELLOW)}: {warn}")

        # Проблемные строки
        if "null_id_sample" in s:
            print(f"       пример строк с null ID: {s['null_id_sample']}")

    # Итог
    print()
    print(_color("─" * 60, BOLD))
    summary = f"  Итого: {passed} OK  |  {warned} с предупреждениями  |  {failed} FAIL  |  всего {total}"
    color = GREEN if failed == 0 else RED
    print(_color(summary, color))
    print(_color("─" * 60, BOLD))
    print()

    return failed == 0


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ETL Parquet outputs")
    parser.add_argument(
        "--verbose", "--fix",
        action="store_true",
        help="Показать примеры проблемных строк",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Проверить только указанные файлы (по имени, например: cast genres)",
    )
    args = parser.parse_args()

    specs = SPECS
    if args.files:
        specs = [s for s in SPECS if s.path.stem in args.files]
        if not specs:
            print(f"Файлы не найдены среди известных: {args.files}")
            return

    results = [check_file(spec, verbose=args.verbose) for spec in specs]
    all_ok = print_report(results)

    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
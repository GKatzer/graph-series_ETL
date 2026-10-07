# Data model

The files the pipeline produces and how they map to the graph and the vector index. Column names and types were read from the local 50-series sample (`data/raw`, `data/processed`, git-ignored) and checked against the code that writes them (`tmdb_fetcher.py`, `embeddings.py`). Row counts below are those of the **sample**, to show shape; a full run has hundreds of thousands of rows per file (the id file alone has 232,755).

**Key:** the TMDB id of a series (`tmdb_id`, called `series_id` in the relation files) is the key everywhere: the Parquet files, the Neo4j `Series` node and the Qdrant point id.

## Raw files (`data/raw/`)

| File | Columns (type) | Sample rows |
|---|---|---|
| `tmdb_series_ids.parquet` | `tmdb_id` (int64), `name` (str), `popularity` (float64) | 232,755 |
| `tmdb_series.parquet` | `tmdb_id` (int64), `name`, `original_name`, `imdb_id`, `wikidata_id` (str), `tvdb_id` (float64), `start_year`, `end_year` (float64), `status`, `type` (str), `in_production` (bool), `original_language` (str), `season_count`, `episode_count` (int64), `episode_runtime`, `popularity`, `vote_average` (float64), `vote_count` (int64), `us_content_rating`, `overview`, `overview_source`, `tagline`, `homepage`, `poster_path` (str) | 50 |
| `tmdb_genres.parquet` | `series_id`, `genre_id` (int64), `genre_name` (str) | 85 |
| `tmdb_keywords.parquet` | `series_id`, `keyword_id` (int64), `keyword_name` (str) | 172 |
| `tmdb_networks.parquet` | `series_id`, `network_id` (int64), `network_name`, `country` (str) | 55 |
| `tmdb_countries.parquet` | `series_id` (int64), `code` (str, ISO 3166-1) | 33 |
| `tmdb_languages.parquet` | `series_id` (int64), `code` (str, ISO 639-1) | 52 |
| `tmdb_cast.parquet` | `series_id`, `person_id` (int64), `person_name` (str), `order`, `episode_count` (int64) | 5,708 |
| `tmdb_creators.parquet` | `series_id`, `person_id` (int64), `person_name` (str) | 63 |
| `tmdb_directors.parquet` | `series_id`, `person_id` (int64), `person_name` (str), `episode_count` (int64) | 247 |
| `tmdb_failed.parquet` | `tmdb_id` (int64): ids whose requests ran out of retries; exists only after such a run | none in the sample |

`tvdb_id`, `start_year` and `end_year` are floats because they can be missing. `overview_source` is `tmdb`, or `wikipedia` for rows replaced by the [Wikipedia fallback](pipeline.md#3-wikipedia-fallback) (42 and 8 in the sample). Text defaults are empty strings and count defaults are `0` (see [`examples/parse_demo.out`](examples/parse_demo.out) for one parsed response, synthetic).

## Processed file (`data/processed/`)

| File | Columns | Sample rows |
|---|---|---|
| `tmdb_embeddings.parquet` | `tmdb_id` (int64), `embedding` (list of 384 float32) | 50 |

## Where each file ends up

| Source file | Qdrant | Neo4j |
|---|---|---|
| `tmdb_series` + `tmdb_embeddings` | one point per embedded series: id `tmdb_id`, payload `{tmdb_id, name, overview, overview_source, imdb_id, start_year}` | `Series` node for series with `vote_count >= 2` |
| `tmdb_cast` | | `Person` nodes, `ACTED_IN` edges (`order` and `episode_count` are not loaded) |
| `tmdb_creators` | | `Person`, `CREATED` |
| `tmdb_directors` | | `Person`, `DIRECTED` (`episode_count` not loaded) |
| `tmdb_genres` / `tmdb_keywords` | | `Genre` / `Keyword` nodes, `HAS_GENRE` / `HAS_KEYWORD` |
| `tmdb_networks` | | `Network` (with `country`), `AIRED_ON` |
| `tmdb_countries` / `tmdb_languages` | | `Country` / `Language` nodes (code only), `PRODUCED_IN` / `HAS_LANGUAGE` |

## Two stores, two coverages

The vector index holds every series that has an embedding (a non-empty overview); the graph only series with at least two votes. The documentation of the project states about 230,000 series fetched (the local id snapshot has 232,755), about 211,000 in the index and about 56,000 in the graph; the last two are not recounted here. The API returns only series present in both, see [`graph-series_backend`](https://github.com/GKatzer/graph-series_backend).

## Graph schema

The graph schema (constraints and indexes) is **not** created by this repository: it is `scripts/schema_init.cypher` in `graph-series_backend`, which must be re-applied after a full reload, otherwise the full-text endpoints fail.

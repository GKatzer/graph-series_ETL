# Design decisions

The decisions that shape the pipeline, with the reason where the code or its comments give one. Where the code does not say why, the entry says what the choice buys and is marked *(inferred from the code)*. Nothing here is taken from old design documents.

**Contents:** [TMDB instead of Wikidata](#tmdb-as-the-source-instead-of-wikidata) · [One request per series](#one-request-per-series-and-english-only) · [Repairing thin descriptions](#repairing-thin-descriptions-from-wikipedia-offline-translation) · [Embedding input](#embedding-input-title-overview-keywords-no-query-instruction) · [A wider index than the graph](#the-vector-index-is-wider-than-the-graph) · [Keys and edges](#keys-and-edges-follow-the-backends-contract) · [Idempotent loads](#idempotent-loads-merge-and-upsert) · [Rate limiting and retries](#rate-limiting-retries-and-the-failed-list) · [Guards](#guards-against-small-data) · [Evaluation design](#the-evaluation-separates-ambiguous-titles) · [Changed or dropped](#changed-or-dropped-along-the-way)

## TMDB as the source instead of Wikidata

The pipeline was rebuilt around the TMDB API (git history: "Rebuild ETL as a TMDB pipeline, replacing Wikidata"). TMDB provides, per series, exactly what the graph needs (cast, creators, genres, keywords, networks, countries, languages, ratings, an overview and a poster path) with a stable numeric id, and its daily ID export lists every known series. Wikidata remains in use only as a bridge: TMDB returns the `wikidata_id`, which the Wikipedia fallback uses to find articles.

## One request per series, and English only

`GET /tv/{id}` with `append_to_response=external_ids,keywords,aggregate_credits,content_ratings` returns details, external ids, keywords, aggregate credits and content ratings in a single call, so about 230,000 series cost about 230,000 requests, not five times that. `aggregate_credits` (not plain credits) gives each person's total episode count across roles. All data is requested in English (`language=en-US`) because TMDB's coverage is best there and the embedding model is English-only; the descriptions that stay thin are repaired afterwards (next section). The rate limit is set to 30 requests per second, with headroom under TMDB's unofficial ceiling of about 40 (comment in `tmdb_fetcher.py`).

## Repairing thin descriptions from Wikipedia, offline translation

Many lesser-known series have a one-line overview on TMDB, and an embedding of a title and one line carries little meaning. For overviews under 200 characters (the code comment says "fewer than about four sentences") the pipeline finds the series' Wikipedia article through the Wikidata sitelinks, preferring the English article, then the article in the show's original language, then any of 31 languages the translation model supports. Non-English text is translated with NLLB-200 (distilled, 600 M parameters) **locally**: no API key, no per-request cost, and the model is downloaded once and cached. The cost is a GPU-hungry step; the default device is `cuda` and the CPU is much slower. Translated text loses some meaning, which the README lists as a limitation. The `overview_source` column (`tmdb` or `wikipedia`) records which rows were replaced, and the replaced TMDB text is not kept.

## Embedding input: title, overview, keywords; no query instruction

The text is `"{name}. {overview} Keywords: …"`, encoded by `BAAI/bge-small-en-v1.5` (384 dimensions) and L2-normalised so that cosine similarity is a dot product. BGE is trained for asymmetric retrieval: queries carry the instruction `Represent this sentence for searching relevant passages: ` and passages do not. Documents are therefore embedded **without** it, and the instruction is added to queries by the inference service in `graph-series_ml` (and by `eval_retrieval.py`, which has to mimic production). Keywords are appended, which adds theme words that an overview may lack *(inferred from the code)*. A series without an overview is not embedded at all.

## The vector index is wider than the graph

Embeddings are computed for every series with an overview, but the graph loads only series with at least two votes (`MIN_VOTE_COUNT = 2`, the code comment calls it the same idea as an earlier `top_series_ids` list). The threshold keeps series that someone has rated and leaves out the long tail of entries with little metadata, which would add many near-empty nodes *(inferred from the code)*. The consequence is that the vector index is about four times larger than the graph; the backend over-fetches from Qdrant and keeps only series that exist in the graph, so every search result can be opened as a graph. *(The ratio, about 211,000 against about 56,000, comes from the repository's comments and the web app's documentation and was not recounted here.)*

## Keys and edges follow the backend's contract

The key is the TMDB id everywhere: the Qdrant point id equals `tmdb_id` because the backend fetches a series' own vector with `retrieve(ids=[tmdb_id])` (docstring of `qdrant_loader.py`). `Series` nodes are keyed by `tmdb_id`, people by `person_id`, countries and languages by their ISO code. Edges have **no properties**, because the backend does not read any (comment in `neo4j_loader.py`), which is also why `order` and `episode_count` stay in the Parquet files. The schema contract (`schema_init.cypher`, the routers) lives in `graph-series_backend`, and this repository follows it instead of defining its own.

## Idempotent loads: MERGE and upsert

The Neo4j loader uses `UNWIND $rows ... MERGE` in batches of 1,000 (nodes first, then edges between series in the graph), and the Qdrant loader upserts by id in batches of 256. Rerunning either converges to the same state and never duplicates, which is what makes partial reruns and corrections cheap. The loaders never delete: there is no wipe, so removing a series from the graph is a manual operation. *(The batch sizes are in the code; the reason for choosing them is not stated.)*

## Rate limiting, retries and the failed list

All HTTP clients share one token-bucket limiter (the same class in the TMDB and Wikipedia modules) so that many workers never exceed the global rate. A 404 means the series was deleted since the export and is not an error; a 429 backs off for 10, 30, 60 and 120 seconds; a server error or a timeout waits and retries; an exhausted series is recorded in `tmdb_failed.parquet` so that `--retry` repeats only those. Series and relations are merged into the existing files and de-duplicated on the series id, so a second run extends the data instead of replacing it.

## Guards against small data

Two checks exist because a leftover smoke-test dataset is easy to mistake for a real one. `validate.py` has a minimum row count per file (150,000 series, 100,000 cast rows, and so on) and fails below it, so a 50-series sample fails on purpose. `eval_retrieval.py` refuses a ground truth of fewer than 1,000 series unless `--allow-small` is given; this was added after the script had quietly accepted a small local file ("fail loudly instead of silently on tiny ground truth" in the history).

## The evaluation separates ambiguous titles

When the target is "the series with this title", a query for a title shared by several series has no single right answer, and counting those as misses understates retrieval quality. The report therefore flags a query as ambiguous when its title occurs more than once in the **whole catalogue** (the namesake is rarely in the sample itself), and gives each metric with and without those queries. The history shows the same step ("separate ambiguous-title collisions from real misses").

## Changed or dropped along the way

- **Wikidata as the primary source** was replaced by TMDB.
- **There is no wipe option** in `neo4j_loader.py` (an old docstring mention was removed); the documented practice is to re-apply the schema after a full reload.
- **`--source auto` of the evaluation** can silently use a local file; it is now guarded by the small-data check, and `--source qdrant` is the documented way to run it.
- **`pipeline.py --resume`** was meant to skip finished steps; it used to skip partial results; now only discover is skipped and the other steps resume by themselves (see [pipeline.md](pipeline.md#known-problems)).

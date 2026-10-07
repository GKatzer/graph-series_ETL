"""Shows what tmdb_fetcher.parse_series / parse_relations do, on a small SYNTHETIC response shaped like
TMDB's  GET /tv/{id}?append_to_response=external_ids,keywords,aggregate_credits,content_ratings.
All values below are invented for illustration; no network access and no API key are needed.

    python docs/examples/parse_demo.py          # from the repository root
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tmdb_fetcher import parse_relations, parse_series  # noqa: E402

response = {
    "id": 424242, "name": "Example Show", "original_name": "Exemple",
    "first_air_date": "2019-03-01", "last_air_date": "2021-06-30", "status": "Ended", "type": "Scripted",
    "in_production": False, "original_language": "fr", "number_of_seasons": 2, "number_of_episodes": 16,
    "episode_run_time": [45], "popularity": 12.5, "vote_average": 7.8, "vote_count": 120,
    "overview": "A short invented overview.", "tagline": "", "homepage": None, "poster_path": "/example.jpg",
    "external_ids": {"imdb_id": "tt0000000", "wikidata_id": "Q0", "tvdb_id": 1},
    "content_ratings": {"results": [{"iso_3166_1": "DE", "rating": "12"}, {"iso_3166_1": "US", "rating": "TV-14"}]},
    "genres": [{"id": 18, "name": "Drama"}],
    "keywords": {"results": [{"id": 7, "name": "invented"}]},
    "networks": [{"id": 49, "name": "Example TV", "origin_country": "FR"}],
    "production_countries": [{"iso_3166_1": "FR"}, {"iso_3166_1": "BE"}],
    "languages": ["fr", "en"],
    "created_by": [{"id": 1, "name": "Creator One"}],
    "aggregate_credits": {
        "cast": [{"id": 2, "name": "Actor Two", "order": 0,
                  "roles": [{"episode_count": 10}, {"episode_count": 6}], "total_episode_count": 16}],
        "crew": [{"id": 3, "name": "Crew Three",
                  "jobs": [{"job": "Director", "episode_count": 4}, {"job": "Writer", "episode_count": 8}]}],
    },
}

print(json.dumps(parse_series(response), indent=2, ensure_ascii=False))
print(json.dumps(parse_relations(response), indent=2, ensure_ascii=False))

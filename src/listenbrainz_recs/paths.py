"""Filesystem locations for raw dumps, intermediate data, and built models.

All paths are relative to the project root, which defaults to the repository
checkout and can be overridden with the LISTENBRAINZ_ROOT environment variable.
"""

import os
from pathlib import Path

ROOT = Path(os.environ.get("LISTENBRAINZ_ROOT", Path(__file__).resolve().parents[2]))

DUMP_DIR = ROOT / "listenbrainz-statistics-dump-20260915-000002"
ARTISTS_ALL_TIME = DUMP_DIR / "lbdump" / "statistics" / "artists_all_time.jsonl"
MUSICBRAINZ_DUMP = ROOT / "data" / "musicbrainz" / "mbdump"

DATA_DIR = ROOT / "data"
INTERACTIONS = DATA_DIR / "user_artist_interactions"
ARTIST_IDS = DATA_DIR / "artist_ids"
ARTIST_FACTORS = DATA_DIR / "artist_factors"
USER_FACTORS = DATA_DIR / "user_factors"
ARTIST_TAGS = DATA_DIR / "artist_tags"
ARTIST_NAMES_CACHE = DATA_DIR / "artist_names.parquet"

MODELS_DIR = ROOT / "models"
ALS_MODEL = MODELS_DIR / "als_64"
FAISS_INDEX = MODELS_DIR / "artists.faiss"
ARTISTS_DB = MODELS_DIR / "artists.sqlite"
UMAP_COORDS = MODELS_DIR / "artist_umap.npy"
ARTIST_MAP = MODELS_DIR / "artist_map.json"

RESULTS_DIR = ROOT / "results"

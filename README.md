# listenbrainz-recs

Musical artist recommendations via collaborative filtering on the
[ListenBrainz](https://listenbrainz.org) statistics dump. An implicit-feedback ALS
model (Spark) learns 64-dimensional artist vectors from users' all-time listen
counts; a FAISS index serves nearest-neighbor "similar artists", and a small
FastAPI web app provides search, recommendations, and a UMAP map of artist space.
MusicBrainz genre tags are used for display and for reranking experiments.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a Java runtime (for PySpark).

```sh
uv sync
```

### Data

Large inputs and outputs live in gitignored directories at the repo root
(override the root with `LISTENBRAINZ_ROOT`):

| Path | Contents |
| --- | --- |
| `listenbrainz-statistics-dump-20260915-000002/` | Extracted [ListenBrainz statistics dump](https://data.metabrainz.org/pub/musicbrainz/listenbrainz/) (uses `lbdump/statistics/artists_all_time.jsonl`) |
| `data/musicbrainz/mbdump/` | Extracted [MusicBrainz database dump](https://data.metabrainz.org/pub/musicbrainz/data/fullexport/) (`artist`, `artist_tag`, `tag`, `genre` tables) |
| `data/` | Intermediate Parquet tables written by the pipeline |
| `models/` | ALS model, FAISS index, SQLite metadata, UMAP coordinates |

## Pipeline

Run each step in order:

```sh
uv run python -m listenbrainz_recs.pipeline.build_matrix            # dump -> user/artist interactions
uv run python -m listenbrainz_recs.pipeline.train_als               # ALS model + artist/user factors
uv run python -m listenbrainz_recs.pipeline.build_artist_tags       # MusicBrainz tags/genres per artist
uv run python -m listenbrainz_recs.pipeline.build_index             # FAISS index + SQLite autocomplete
uv run python -m listenbrainz_recs.pipeline.update_artist_metadata  # popularity + genres into SQLite
uv run python -m listenbrainz_recs.pipeline.build_umap              # 2D artist map (optional)
```

## Web app

```sh
uv run uvicorn listenbrainz_recs.app.main:app --reload
```

Then open <http://127.0.0.1:8000/> (recommendations) or <http://127.0.0.1:8000/map>.

## Experiments

Command-line tools for exploring the model, in `listenbrainz_recs.experiments`:

| Module | Purpose |
| --- | --- |
| `similar_artists "Henry Threadgill"` | Nearest neighbors by ALS cosine similarity |
| `rerank_genres "Rabih Abou-Khalil"` | Compare raw ALS with genre-aware reranking (`--method confidence\|blend`, `--coherence`) |
| `compare_similarities` | Raw vs. centered / PCA-adjusted vector similarities |
| `evaluate_artists export ...` / `score` | Blinded relevance judgments and NDCG for raw vs. reranked lists |
| `inspect_matrix`, `count_artists` | Dataset summary statistics |

Run any of them with `uv run python -m listenbrainz_recs.experiments.<module>`.
Hand-labeled relevance judgments are in `results/`.

## Development

```sh
uv run ruff check
uv run ruff format
```

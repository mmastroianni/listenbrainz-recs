"""FastAPI serving layer for the ListenBrainz ALS artist recommender.

Run with:
    uv run uvicorn listenbrainz_recs.app.main:app --reload
"""

import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from listenbrainz_recs.paths import ARTIST_MAP, ARTISTS_DB, FAISS_INDEX, UMAP_COORDS
from listenbrainz_recs.text import normalize_name

STATIC_PATH = Path(__file__).resolve().parent / "static"


def connect() -> sqlite3.Connection:
    db = sqlite3.connect(f"file:{ARTISTS_DB}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not FAISS_INDEX.is_file() or not ARTISTS_DB.is_file():
        raise RuntimeError("Missing index artifacts. Run the build_index pipeline step first.")
    index = faiss.read_index(str(FAISS_INDEX))
    with connect() as db:
        count = db.execute("SELECT COUNT(*) FROM artists").fetchone()[0]
    if count != index.ntotal:
        raise RuntimeError(f"FAISS/SQLite count mismatch: {index.ntotal} vs {count}")
    with connect() as db:
        columns = {r[1] for r in db.execute("PRAGMA table_info(artists)")}
        genres_ready = db.execute(
            "SELECT name FROM sqlite_master WHERE name='artist_genres'"
        ).fetchone()
        if not {"listeners", "total_listens"}.issubset(columns) or not genres_ready:
            raise RuntimeError(
                "Run the update_artist_metadata pipeline step before starting the server"
            )
    app.state.index = index
    yield


app = FastAPI(title="ListenBrainz Artist Recommender", lifespan=lifespan)


def genres_for(db, mbids, top=4):
    """Fetch strongest MusicBrainz genres in one SQL query."""
    if not mbids:
        return {}
    placeholders = ",".join("?" for _ in mbids)
    rows = db.execute(
        f"""
        SELECT artist_mbid, tag FROM (
            SELECT artist_mbid, tag,
                   row_number() OVER (PARTITION BY artist_mbid ORDER BY votes DESC, tag) AS rn
            FROM artist_genres WHERE artist_mbid IN ({placeholders})
        ) WHERE rn <= ? ORDER BY artist_mbid, rn
    """,
        (*mbids, top),
    ).fetchall()
    result = {mbid: [] for mbid in mbids}
    for row in rows:
        result[row["artist_mbid"]].append(row["tag"])
    return result


@app.get("/api/suggest")
def suggest(q: str = Query(min_length=1), limit: int = Query(10, ge=1, le=50)):
    normalized = normalize_name(q)
    tokens = normalized.split()
    if not tokens:
        return []
    expression = " AND ".join(f'"{token}"*' for token in tokens)
    # FTS finds candidates; SQL then prioritizes exact/prefix names and listener counts.
    # The FTS expression uses normalized alphanumeric tokens only.
    with connect() as db:
        rows = db.execute(
            """
            SELECT a.artist_mbid, a.name, a.listeners
            FROM artist_search s JOIN artists a ON a.faiss_row = s.rowid
            WHERE artist_search MATCH ?
            ORDER BY
                CASE
                    WHEN a.search_name = ? THEN 0
                    WHEN substr(a.search_name, 1, length(?)+1) = ? || ' ' THEN 1
                    WHEN substr(a.search_name, 1, length(?)) = ? THEN 2
                    ELSE 3
                END,
                a.listeners DESC, a.total_listens DESC, length(a.name), a.name
            LIMIT ?
        """,
            (expression, normalized, normalized, normalized, normalized, normalized, limit),
        ).fetchall()
    return [
        {"mbid": row["artist_mbid"], "name": row["name"], "listeners": row["listeners"]}
        for row in rows
    ]


@app.get("/api/similar/{mbid}")
def similar(request: Request, mbid: str, limit: int = Query(20, ge=1, le=100)):
    index = request.app.state.index
    with connect() as db:
        seed = db.execute(
            "SELECT faiss_row, artist_mbid, name FROM artists WHERE artist_mbid = ?",
            (mbid,),
        ).fetchone()
        if seed is None:
            raise HTTPException(status_code=404, detail="Artist not found in trained ALS model")
        seed_row = int(seed["faiss_row"])
        query_vector = np.empty((1, index.d), dtype=np.float32)
        index.reconstruct(seed_row, query_vector[0])
        scores, positions = index.search(query_vector, min(index.ntotal, limit + 1))
        candidates = [
            (int(pos), float(score))
            for pos, score in zip(positions[0], scores[0], strict=True)
            if pos >= 0 and int(pos) != seed_row
        ][:limit]
        results = []
        if candidates:
            placeholders = ",".join("?" for _ in candidates)
            rows = db.execute(
                "SELECT faiss_row, artist_mbid, name FROM artists "
                f"WHERE faiss_row IN ({placeholders})",
                tuple(pos for pos, _ in candidates),
            ).fetchall()
            by_row = {int(row["faiss_row"]): row for row in rows}
            if len(by_row) != len(candidates):
                raise HTTPException(status_code=500, detail="FAISS/SQLite row mapping mismatch")
            results = [
                {
                    "mbid": by_row[pos]["artist_mbid"],
                    "name": by_row[pos]["name"],
                    "als_score": score,
                }
                for pos, score in candidates
            ]
        all_mbids = [mbid] + [result["mbid"] for result in results]
        genre_map = genres_for(db, all_mbids)
        for result in results:
            result["genres"] = genre_map.get(result["mbid"], [])
    return {
        "seed": {
            "mbid": seed["artist_mbid"],
            "name": seed["name"],
            "genres": genre_map.get(mbid, []),
        },
        "results": results,
    }


app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_PATH / "index.html")


# Artist map: independent of recommendation ranking.


@app.get("/map", include_in_schema=False)
def map_page():
    return FileResponse(STATIC_PATH / "map.html")


@app.get("/api/map")
def map_points():
    if not ARTIST_MAP.is_file():
        raise HTTPException(404, "Run the build_umap pipeline step first")
    return FileResponse(ARTIST_MAP, media_type="application/json")


@app.get("/api/map/artist/{mbid}")
def map_artist(request: Request, mbid: str, limit: int = Query(20, ge=1, le=100)):
    if not UMAP_COORDS.is_file():
        raise HTTPException(404, "Run the build_umap pipeline step first")
    if not hasattr(request.app.state, "map_coords"):
        request.app.state.map_coords = np.load(UMAP_COORDS, mmap_mode="r")
    coords = request.app.state.map_coords
    index = request.app.state.index
    with connect() as db:
        seed = db.execute(
            "SELECT faiss_row, name, artist_mbid FROM artists WHERE artist_mbid=?", (mbid,)
        ).fetchone()
        if seed is None:
            raise HTTPException(404, "Artist not found")
        row = int(seed["faiss_row"])
        query = np.empty((1, index.d), dtype=np.float32)
        index.reconstruct(row, query[0])
        scores, positions = index.search(query, min(index.ntotal, limit + 1))
        neighbors = [
            (int(p), float(s))
            for p, s in zip(positions[0], scores[0], strict=True)
            if int(p) >= 0 and int(p) != row
        ][:limit]
        ids = [row] + [p for p, _ in neighbors]
        placeholders = ",".join("?" for _ in ids)
        artists = {
            int(r["faiss_row"]): dict(r)
            for r in db.execute(
                "SELECT faiss_row, artist_mbid, name FROM artists "
                f"WHERE faiss_row IN ({placeholders})",
                ids,
            )
        }

    def point(i, score=None):
        a = artists[i]
        return {
            "mbid": a["artist_mbid"],
            "name": a["name"],
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "als_score": score,
        }

    return {"seed": point(row), "neighbors": [point(i, s) for i, s in neighbors]}


@app.get("/api/map/neighborhood")
def map_neighborhood(request: Request, x: float, y: float, count: int = Query(250, ge=30, le=1000)):
    """Describe the closest projected artists to a map location.

    UMAP distances are approximate. Counts are for this local sample, not
    for a global genre census. Uses the full coordinate array, not only dots.
    """
    if not UMAP_COORDS.is_file():
        raise HTTPException(404, "Run the build_umap pipeline step first")
    if not np.isfinite(x) or not np.isfinite(y):
        raise HTTPException(422, "Coordinates must be finite")
    if not hasattr(request.app.state, "map_coords"):
        request.app.state.map_coords = np.load(UMAP_COORDS, mmap_mode="r")
    coords = request.app.state.map_coords
    if len(coords) != request.app.state.index.ntotal:
        raise HTTPException(500, "UMAP/FAISS index size mismatch")
    dist2 = (coords[:, 0] - x) ** 2 + (coords[:, 1] - y) ** 2
    k = min(count, len(coords))
    nearest = np.argpartition(dist2, k - 1)[:k]
    ids = [int(i) for i in nearest]
    placeholders = ",".join("?" for _ in ids)
    with connect() as db:
        rows = db.execute(
            f"""SELECT faiss_row, artist_mbid, name, listeners
            FROM artists WHERE faiss_row IN ({placeholders})""",
            ids,
        ).fetchall()
        if len(rows) != k:
            raise HTTPException(500, "UMAP/SQLite row mapping mismatch")
        popular = sorted(rows, key=lambda r: (-int(r["listeners"] or 0), r["name"]))[:12]
        # Each artist can have several genre tags. For the neighborhood guide,
        # use the highest-voted tag per artist so each artist contributes once.
        genre_rows = db.execute(
            f"""
            SELECT tag, COUNT(*) AS artists FROM (
                SELECT g.artist_mbid, g.tag,
                    ROW_NUMBER() OVER (PARTITION BY g.artist_mbid
                        ORDER BY g.votes DESC, g.tag) AS rn
                FROM artist_genres g JOIN artists a ON a.artist_mbid = g.artist_mbid
                WHERE a.faiss_row IN ({placeholders})
            ) WHERE rn = 1 GROUP BY tag ORDER BY artists DESC, tag LIMIT 8
        """,
            ids,
        ).fetchall()
    return {
        "sample_size": k,
        "genres": [{"name": r["tag"], "count": r["artists"]} for r in genre_rows],
        "artists": [
            {"mbid": r["artist_mbid"], "name": r["name"], "listeners": r["listeners"]}
            for r in popular
        ],
    }

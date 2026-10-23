"""Step 6 (optional): build the 2D artist map shown at /map.

Fits UMAP on a genre-stratified sample of artists, then projects every trained
artist. Reads only the FAISS index and SQLite metadata; no Spark needed.
"""

import argparse
import json
import sqlite3
from collections import defaultdict

import faiss
import numpy as np
import umap

from listenbrainz_recs.paths import ARTIST_MAP, ARTISTS_DB, FAISS_INDEX, UMAP_COORDS

GENRES = {
    "jazz": ("jazz", "bebop", "free improvisation", "avant-garde jazz", "fusion", "swing"),
    "classical": (
        "classical",
        "baroque",
        "early music",
        "renaissance",
        "romantic",
        "opera",
        "contemporary classical",
        "modern classical",
        "chamber music",
    ),
    "world": (
        "world",
        "folk",
        "traditional",
        "afrobeat",
        "flamenco",
        "arabic",
        "turkish",
        "indian classical",
        "qawwali",
        "fado",
        "klezmer",
        "balkan",
    ),
    "electronic": ("electronic", "ambient", "techno", "house", "idm", "drum and bass"),
    "rock": ("rock", "punk", "post-punk", "indie rock", "alternative rock", "progressive rock"),
    "metal": ("metal", "black metal", "death metal", "doom metal"),
    "hip-hop": ("hip hop", "hip-hop", "rap", "trap"),
    "pop": ("pop", "synthpop", "dance-pop", "k-pop"),
    "other": (),
}
# Prioritize specific tags before broad tags; one display group per artist.
PRIORITY = ("metal", "hip-hop", "jazz", "world", "electronic", "classical", "rock", "pop")


def categorize(tags):
    for group in PRIORITY:
        terms = GENRES[group]
        if any(any(term in tag for term in terms) for tag in tags):
            return group
    return "other"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=15000)
    parser.add_argument("--neighbors", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    print("Reading FAISS vectors and SQLite metadata...", flush=True)
    index = faiss.read_index(str(FAISS_INDEX))
    if not isinstance(index, faiss.IndexFlatIP):
        raise SystemExit("Expected IndexFlatIP from the build_index step")
    vectors = np.empty((index.ntotal, index.d), dtype=np.float32)
    index.reconstruct_n(0, index.ntotal, vectors)
    db = sqlite3.connect(ARTISTS_DB)
    db.row_factory = sqlite3.Row
    artists = db.execute(
        "SELECT faiss_row, artist_mbid, name, listeners FROM artists ORDER BY faiss_row"
    ).fetchall()
    if len(artists) != index.ntotal or any(int(a["faiss_row"]) != i for i, a in enumerate(artists)):
        raise SystemExit("SQLite FAISS row mapping mismatch")
    tags = defaultdict(list)
    for row in db.execute("SELECT artist_mbid, tag FROM artist_genres WHERE votes > 0"):
        tags[row["artist_mbid"]].append(row["tag"].lower())
    db.close()
    groups = np.array([categorize(tags[a["artist_mbid"]]) for a in artists])
    popularity = np.array([max(0, int(a["listeners"])) for a in artists], dtype=np.float64)
    # Keep substantial coverage from each represented group, including niche artists.
    buckets = {g: np.flatnonzero(groups == g) for g in GENRES}
    n = min(args.sample, len(artists))
    active = [g for g, ids in buckets.items() if len(ids)]
    min_each = min(800, n // max(1, len(active) * 2))
    selected = []
    for group in active:
        ids = buckets[group]
        k = min(len(ids), min_each)
        # Mixed popularity/random sampling: obscure artists remain eligible.
        weights = np.sqrt(popularity[ids] + 1)
        weights = 0.6 * (weights / weights.sum()) + 0.4 / len(ids)
        selected.extend(rng.choice(ids, size=k, replace=False, p=weights).tolist())
    remaining = n - len(selected)
    if remaining > 0:
        available = np.setdiff1d(np.arange(len(artists)), np.array(selected, dtype=int))
        weights = np.sqrt(popularity[available] + 1)
        weights = 0.5 * weights / weights.sum() + 0.5 / len(available)
        selected.extend(rng.choice(available, size=remaining, replace=False, p=weights).tolist())
    selected = np.array(sorted(selected), dtype=int)
    print(f"Fitting UMAP on {len(selected):,} artists, {index.d} dimensions...", flush=True)
    reducer = umap.UMAP(
        n_neighbors=args.neighbors,
        min_dist=0.12,
        metric="cosine",
        random_state=args.seed,
        low_memory=True,
        n_jobs=1,
    )
    reducer.fit(vectors[selected])
    print(f"Projecting all {index.ntotal:,} artists (may take a while)...", flush=True)
    coords = np.empty((index.ntotal, 2), dtype=np.float32)
    # Preserve exact fitted positions for sampled artists.
    for start in range(0, index.ntotal, 10000):
        stop = min(index.ntotal, start + 10000)
        coords[start:stop] = reducer.transform(vectors[start:stop])
        print(f"  {stop:,}/{index.ntotal:,}", flush=True)
    coords[selected] = reducer.embedding_.astype(np.float32)
    UMAP_COORDS.parent.mkdir(parents=True, exist_ok=True)
    np.save(UMAP_COORDS, coords)
    # Keep map responsive: sampled points plus all artists available via API overlay.
    points = [
        {
            "row": int(i),
            "mbid": artists[i]["artist_mbid"],
            "name": artists[i]["name"],
            "genre": str(groups[i]),
            "listeners": int(popularity[i]),
            "x": round(float(coords[i, 0]), 3),
            "y": round(float(coords[i, 1]), 3),
        }
        for i in selected
    ]
    ARTIST_MAP.write_text(json.dumps(points, ensure_ascii=False, separators=(",", ":")))
    print(f"Saved {UMAP_COORDS} and {ARTIST_MAP} ({len(points):,} displayed points)", flush=True)


if __name__ == "__main__":
    main()

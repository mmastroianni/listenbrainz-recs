"""Step 4: build the exact FAISS cosine index and SQLite artist autocomplete.

FAISS row i and SQLite artists.faiss_row i refer to the same artist.
"""

import argparse
import sqlite3

import faiss
import numpy as np

from listenbrainz_recs.artists import load_artist_factors
from listenbrainz_recs.paths import ARTISTS_DB, FAISS_INDEX
from listenbrainz_recs.text import normalize_name


def build_sqlite(path, artists):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.executescript("""
            DROP TABLE IF EXISTS artist_search;
            DROP TABLE IF EXISTS artists;
            CREATE TABLE artists (
                faiss_row INTEGER PRIMARY KEY,
                artist_id INTEGER NOT NULL UNIQUE,
                artist_mbid TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                search_name TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE artist_search USING fts5(
                search_name, content='artists', content_rowid='faiss_row',
                tokenize='unicode61'
            );
        """)
        db.executemany("INSERT INTO artists VALUES (?, ?, ?, ?, ?)", artists)
        db.execute("INSERT INTO artist_search(artist_search) VALUES ('rebuild')")
        db.execute("CREATE INDEX artists_mbid_idx ON artists(artist_mbid)")


def suggest(db, query, limit=10):
    """Prefix-match each normalized token, including infix artist-name tokens."""
    tokens = normalize_name(query).split()
    if not tokens:
        return []
    # All tokens are alphanumeric after normalization; no FTS syntax injection.
    expression = " AND ".join(f'"{token}"*' for token in tokens)
    return db.execute(
        """
        SELECT a.artist_mbid, a.name, a.faiss_row
        FROM artist_search s JOIN artists a ON a.faiss_row = s.rowid
        WHERE artist_search MATCH ?
        ORDER BY length(a.name), a.name LIMIT ?
        """,
        (expression, limit),
    ).fetchall()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-query", default="abou khalil")
    args = parser.parse_args()

    factors = load_artist_factors("BuildArtistIndex", normalize=False)
    vectors = factors.vectors
    if not np.isfinite(vectors).all():
        raise ValueError("Invalid factor matrix")
    if np.any(np.linalg.norm(vectors, axis=1) <= 1e-12):
        raise ValueError("Zero-norm factor vectors cannot be cosine indexed")
    faiss.normalize_L2(vectors)

    records = []
    for position, artist_id in enumerate(factors.ids):
        artist_id = int(artist_id)
        if artist_id not in factors.by_id:
            raise ValueError(f"No MBID mapping for factor artist_id={artist_id}")
        mbid, name = factors.by_id[artist_id]
        if not mbid:
            raise ValueError(f"Missing MBID for artist_id={artist_id}")
        name = name or mbid
        records.append((position, artist_id, mbid, name, normalize_name(name)))

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    FAISS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX))
    build_sqlite(ARTISTS_DB, records)

    with sqlite3.connect(ARTISTS_DB) as db:
        matches = suggest(db, args.test_query)
        print(f"Autocomplete for {args.test_query!r}:")
        for mbid, name, row in matches:
            print(f"  {name} [{mbid}] (FAISS row {row})")

        # Validate the row-to-artist mapping against an exact FAISS query.
        row = matches[0][2] if matches else 0
        scores, positions = index.search(vectors[row : row + 1], 6)
        print(f"\nNearest neighbors for {records[row][3]}:")
        for score, pos in zip(scores[0], positions[0], strict=True):
            if pos == row or pos < 0:
                continue
            (name,) = db.execute(
                "SELECT name FROM artists WHERE faiss_row=?", (int(pos),)
            ).fetchone()
            print(f"  {name}: {float(score):.4f}")

    print(f"\nIndexed {index.ntotal:,} artists ({index.d} dimensions)")
    print(f"FAISS:  {FAISS_INDEX}")
    print(f"SQLite: {ARTISTS_DB}")


if __name__ == "__main__":
    main()

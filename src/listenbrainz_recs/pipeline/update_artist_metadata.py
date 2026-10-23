"""Step 5: add listener popularity and MusicBrainz genre labels to the SQLite index.

Run after build_index. No ALS retraining or FAISS rebuild required.
"""

import sqlite3

from pyspark.sql import functions as F

from listenbrainz_recs.paths import ARTIST_TAGS, ARTISTS_DB, INTERACTIONS
from listenbrainz_recs.spark import get_spark


def main():
    if not ARTISTS_DB.is_file():
        raise SystemExit(f"Missing {ARTISTS_DB}; run the build_index step first")
    spark = get_spark("ArtistMetadata", log_level="ERROR")
    try:
        print("Computing listener counts and total listens...", flush=True)
        interactions = spark.read.parquet(str(INTERACTIONS))
        # Data contains one row per (user, artist); countDistinct is safe if not.
        popularity = interactions.groupBy("artist_mbid").agg(
            F.countDistinct("user_id").alias("listeners"),
            F.sum("listen_count").alias("total_listens"),
        )
        popularity_rows = [
            (str(r.artist_mbid), int(r.listeners), int(r.total_listens or 0))
            for r in popularity.collect()
        ]
        print("Loading MusicBrainz genre tags...", flush=True)
        tags = (
            spark.read.parquet(str(ARTIST_TAGS))
            .filter(F.col("is_genre") & (F.col("tag_count") > 0))
            .groupBy("artist_mbid", "tag")
            .agg(F.sum("tag_count").alias("votes"))
        )
        tag_rows = [(str(r.artist_mbid), str(r.tag), int(r.votes)) for r in tags.collect()]
    finally:
        spark.stop()

    # SQLite transaction: if anything fails, the previous DB remains usable.
    with sqlite3.connect(ARTISTS_DB) as db:
        db.execute("PRAGMA busy_timeout=10000")
        db.execute("BEGIN IMMEDIATE")
        cols = {row[1] for row in db.execute("PRAGMA table_info(artists)")}
        if "listeners" not in cols:
            db.execute("ALTER TABLE artists ADD COLUMN listeners INTEGER NOT NULL DEFAULT 0")
        if "total_listens" not in cols:
            db.execute("ALTER TABLE artists ADD COLUMN total_listens INTEGER NOT NULL DEFAULT 0")
        db.execute("UPDATE artists SET listeners=0, total_listens=0")
        # TEMP staging table avoids hundreds of thousands of individual updates.
        db.execute(
            "CREATE TEMP TABLE popularity "
            "(mbid TEXT PRIMARY KEY, listeners INTEGER, total_listens INTEGER)"
        )
        db.executemany("INSERT INTO popularity VALUES (?,?,?)", popularity_rows)
        db.execute("""
            UPDATE artists SET
                listeners = COALESCE(
                    (SELECT listeners FROM popularity p WHERE p.mbid = artists.artist_mbid), 0),
                total_listens = COALESCE(
                    (SELECT total_listens FROM popularity p WHERE p.mbid = artists.artist_mbid), 0)
        """)
        db.execute("DROP TABLE IF EXISTS artist_genres")
        db.execute("""CREATE TABLE artist_genres (
            artist_mbid TEXT NOT NULL,
            tag TEXT NOT NULL,
            votes INTEGER NOT NULL,
            PRIMARY KEY(artist_mbid, tag)
        )""")
        db.executemany("INSERT INTO artist_genres VALUES (?,?,?)", tag_rows)
        db.execute("CREATE INDEX artist_genres_top ON artist_genres(artist_mbid, votes DESC)")
        count = db.execute("SELECT COUNT(*) FROM artists WHERE listeners > 0").fetchone()[0]
        tagged = db.execute("SELECT COUNT(DISTINCT artist_mbid) FROM artist_genres").fetchone()[0]
    print(f"Updated {count:,} indexed artists with popularity; {tagged:,} artists have genre tags.")
    print("FAISS index unchanged. Restart uvicorn to use updated metadata.")


if __name__ == "__main__":
    main()

"""Load trained ALS artist factors and resolve artists by name or MBID."""

from dataclasses import dataclass, field

import numpy as np
from pyspark.sql import Row, SparkSession, Window
from pyspark.sql import functions as F

from listenbrainz_recs.paths import ARTIST_FACTORS, ARTIST_IDS, INTERACTIONS
from listenbrainz_recs.spark import get_spark

# Nicknames accepted by resolve_artist in addition to full names and MBIDs.
SHORTCUTS = {
    "braxton": "4e5f99db-d181-4317-9124-602b776f5306",
    "taylor": "0d06bb57-c3b6-4188-9b73-826c41c3e29e",
    "threadgill": "bf3f2266-33e7-42ae-a1f2-4066ded25dca",
    "nichols": "d179c4af-d690-4cdd-a769-ba095b42fa1a",
    "smith": "11ce83c2-1ae1-4958-a28a-b58e61a6fb60",
    "pullen": "7a31f6c3-92cd-4b9d-b929-f168ef12b8cd",
}


def load_artists(spark: SparkSession) -> list[Row]:
    """Return (artist_id, artist_mbid, name) rows for every trained artist.

    Each MBID gets its most frequently listened-to name.
    """
    names = (
        spark.read.parquet(str(INTERACTIONS))
        .groupBy("artist_mbid", "artist_name")
        .agg(F.sum("listen_count").alias("total_listens"))
    )
    window = Window.partitionBy("artist_mbid").orderBy(
        F.desc("total_listens"), F.asc("artist_name")
    )
    names = (
        names.withColumn("rank", F.row_number().over(window))
        .filter(F.col("rank") == 1)
        .select("artist_mbid", F.col("artist_name").alias("name"))
    )
    return (
        spark.read.parquet(str(ARTIST_IDS))
        .join(names, "artist_mbid", "left")
        .select("artist_id", "artist_mbid", "name")
        .collect()
    )


def resolve_artist(query: str, artists: list[Row]) -> int | None:
    """Find an artist_id by nickname, MBID, or name; print candidates if ambiguous."""
    query_lower = query.casefold().strip()
    mbid_to_id = {row.artist_mbid: int(row.artist_id) for row in artists}

    if query_lower in SHORTCUTS:
        return mbid_to_id.get(SHORTCUTS[query_lower])
    if query in mbid_to_id:
        return mbid_to_id[query]

    exact = [row for row in artists if row.name and row.name.casefold() == query_lower]
    if len(exact) == 1:
        return int(exact[0].artist_id)

    matches = exact or [row for row in artists if row.name and query_lower in row.name.casefold()]
    if not matches:
        print(f"No artists found matching {query!r}")
        return None

    print(f"\nArtists matching {query!r}:\n")
    for row in sorted(matches, key=lambda r: (r.name or "").casefold())[:30]:
        print(f"  {row.name:45s} {row.artist_mbid}")
    if len(matches) > 30:
        print(f"\n  ... and {len(matches) - 30} more")
    print("\nUse the full artist name or MBID to select one.")
    return None


def top_positions(scores: np.ndarray, n: int) -> np.ndarray:
    """Indices of the n highest scores, best first."""
    n = min(n, len(scores))
    if n == 0:
        return np.array([], dtype=np.int64)
    positions = np.argpartition(scores, -n)[-n:]
    return positions[np.argsort(scores[positions])[::-1]]


@dataclass
class ArtistFactors:
    """ALS artist vectors sorted by artist_id, with their metadata."""

    artists: list[Row]
    ids: np.ndarray
    vectors: np.ndarray
    by_id: dict[int, tuple[str, str | None]] = field(init=False)

    def __post_init__(self):
        self.by_id = {int(row.artist_id): (row.artist_mbid, row.name) for row in self.artists}

    def position(self, artist_id: int) -> int | None:
        matches = np.flatnonzero(self.ids == artist_id)
        return int(matches[0]) if len(matches) else None

    def mbid(self, position: int) -> str | None:
        return self.by_id.get(int(self.ids[position]), (None, None))[0]

    def label(self, position: int) -> str:
        mbid, name = self.by_id.get(int(self.ids[position]), ("unknown", "unknown"))
        return name or mbid

    def similarities(self, position: int) -> np.ndarray:
        """Cosine similarity of every artist to the one at position (itself excluded)."""
        scores = self.vectors @ self.vectors[position]
        scores[position] = -np.inf
        return scores


def load_artist_factors(app_name: str, *, normalize: bool = True) -> ArtistFactors:
    """Read artist metadata and ALS factors with Spark, then release the session."""
    spark = get_spark(app_name, log_level="ERROR")
    try:
        print("Loading artists and factors...", flush=True)
        artists = load_artists(spark)
        factors = spark.read.parquet(str(ARTIST_FACTORS)).collect()
    finally:
        spark.stop()

    if not factors:
        raise RuntimeError("No ALS factors found")
    factors.sort(key=lambda row: int(row.id))
    ids = np.asarray([row.id for row in factors], dtype=np.int32)
    vectors = np.ascontiguousarray(np.asarray([row.features for row in factors], dtype=np.float32))
    if normalize:
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return ArtistFactors(artists, ids, vectors)

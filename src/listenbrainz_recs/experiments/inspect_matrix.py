"""Print summary statistics for the user-artist interaction table."""

from pyspark.sql import functions as F

from listenbrainz_recs.paths import INTERACTIONS
from listenbrainz_recs.spark import get_spark

JAZZ_ARTISTS = [
    "Henry Threadgill",
    "Anthony Braxton",
    "Cecil Taylor",
    "Herbie Nichols",
    "Don Pullen",
    "Wadada Leo Smith",
]


def artist_stats(df):
    return df.groupBy("artist_mbid", "artist_name").agg(
        F.countDistinct("user_id").alias("listeners"),
        F.sum("listen_count").alias("total_plays"),
    )


def main():
    spark = get_spark("InspectListenBrainz", shuffle_partitions=16)
    df = spark.read.parquet(str(INTERACTIONS))

    summary = df.agg(
        F.count("*").alias("interactions"),
        F.countDistinct("user_id").alias("users"),
        F.countDistinct("artist_mbid").alias("artists"),
    ).first()
    print(f"\nInteractions: {summary['interactions']:,}")
    print(f"Users:        {summary['users']:,}")
    print(f"Artists:      {summary['artists']:,}")

    print("\nTop 20 artists by number of listeners:")
    artist_stats(df).orderBy(F.desc("listeners")).show(20, truncate=False)

    # Spot-check some less mainstream jazz artists.
    print("\nSelected jazz artists:")
    artist_stats(df.filter(F.col("artist_name").isin(JAZZ_ARTISTS))).orderBy("artist_name").show(
        50, truncate=False
    )

    spark.stop()


if __name__ == "__main__":
    main()

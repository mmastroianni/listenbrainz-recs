"""Step 1: explode the ListenBrainz all-time artist statistics into user-artist interactions."""

from pyspark.sql import functions as F
from pyspark.sql import types as T

from listenbrainz_recs.paths import ARTISTS_ALL_TIME, INTERACTIONS
from listenbrainz_recs.spark import get_spark

# Define the schema explicitly to avoid an expensive inference pass.
SCHEMA = T.StructType(
    [
        T.StructField("user_id", T.LongType()),
        T.StructField(
            "data",
            T.ArrayType(
                T.StructType(
                    [
                        T.StructField("artist_mbid", T.StringType()),
                        T.StructField("artist_name", T.StringType()),
                        T.StructField("listen_count", T.LongType()),
                    ]
                )
            ),
        ),
    ]
)


def main():
    spark = get_spark("ListenBrainzArtistInteractions", shuffle_partitions=16)

    # One JSON object per user, holding an array of that user's artists.
    users = spark.read.schema(SCHEMA).json(str(ARTISTS_ALL_TIME))
    interactions = (
        users.select("user_id", F.explode("data").alias("artist"))
        .select(
            "user_id",
            F.col("artist.artist_mbid").alias("artist_mbid"),
            F.col("artist.artist_name").alias("artist_name"),
            F.col("artist.listen_count").alias("listen_count"),
        )
        .filter(
            F.col("user_id").isNotNull()
            & F.col("artist_mbid").isNotNull()
            & (F.col("listen_count") > 0)
        )
    )
    interactions.write.mode("overwrite").parquet(str(INTERACTIONS))
    print(f"Wrote interactions to {INTERACTIONS}")

    spark.stop()


if __name__ == "__main__":
    main()

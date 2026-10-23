"""Step 2: train an implicit-feedback ALS model on user-artist listen counts."""

from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F

from listenbrainz_recs.paths import (
    ALS_MODEL,
    ARTIST_FACTORS,
    ARTIST_IDS,
    INTERACTIONS,
    USER_FACTORS,
)
from listenbrainz_recs.spark import get_spark

MIN_LISTENERS = 5


def main():
    spark = get_spark("ListenBrainzALS")

    print("Loading interactions...", flush=True)
    df = spark.read.parquet(str(INTERACTIONS))

    print(f"Keeping artists with at least {MIN_LISTENERS} listeners...", flush=True)
    valid_artists = (
        df.groupBy("artist_mbid")
        .agg(F.countDistinct("user_id").alias("listeners"))
        .filter(F.col("listeners") >= MIN_LISTENERS)
        .select("artist_mbid")
    )
    df = df.join(valid_artists, "artist_mbid")

    # Assign integer artist IDs without a global window.
    print("Indexing artists...", flush=True)
    indexer = StringIndexer(
        inputCol="artist_mbid",
        outputCol="artist_id",
        handleInvalid="error",
        stringOrderType="alphabetAsc",
    )
    df = indexer.fit(df).transform(df).withColumn("artist_id", F.col("artist_id").cast("int"))
    df.select("artist_mbid", "artist_id").distinct().write.mode("overwrite").parquet(
        str(ARTIST_IDS)
    )

    # Spark ALS requires integer user and item IDs. Log-transform counts to
    # reduce the influence of heavy listeners.
    training = df.select(
        F.col("user_id").cast("int").alias("user"),
        F.col("artist_id").alias("item"),
        F.log1p(F.col("listen_count")).cast("float").alias("rating"),
    )

    print("Starting ALS training...", flush=True)
    als = ALS(
        rank=64,
        maxIter=10,
        regParam=0.1,
        alpha=20.0,
        implicitPrefs=True,
        userCol="user",
        itemCol="item",
        ratingCol="rating",
        coldStartStrategy="drop",
        numUserBlocks=4,
        numItemBlocks=4,
    )
    model = als.fit(training)

    print("Saving model and factors...", flush=True)
    model.write().overwrite().save(str(ALS_MODEL))
    model.itemFactors.write.mode("overwrite").parquet(str(ARTIST_FACTORS))
    model.userFactors.write.mode("overwrite").parquet(str(USER_FACTORS))
    print("Done.", flush=True)

    spark.stop()


if __name__ == "__main__":
    main()

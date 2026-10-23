"""Step 3: join MusicBrainz artist tags (flagging recognized genres) onto trained artists."""

from pyspark.sql import functions as F

from listenbrainz_recs.paths import ARTIST_IDS, ARTIST_TAGS, MUSICBRAINZ_DUMP
from listenbrainz_recs.spark import get_spark


def load_table(spark, name, columns):
    """Load selected columns, given as (index, alias) pairs, from a MusicBrainz TSV dump."""
    df = (
        spark.read.option("sep", "\t")
        .option("header", "false")
        .option("nullValue", "\\N")
        .csv(str(MUSICBRAINZ_DUMP / name))
    )
    return df.select(*[F.col(f"_c{index}").alias(alias) for index, alias in columns])


def main():
    spark = get_spark("BuildArtistTags")

    print("Loading MusicBrainz artists...", flush=True)
    artists = load_table(
        spark, "artist", [(0, "mb_artist_id"), (1, "artist_mbid"), (2, "artist_name")]
    ).withColumn("mb_artist_id", F.col("mb_artist_id").cast("long"))

    print("Loading artist tags...", flush=True)
    artist_tags = (
        load_table(spark, "artist_tag", [(0, "mb_artist_id"), (1, "tag_id"), (2, "tag_count")])
        .withColumn("mb_artist_id", F.col("mb_artist_id").cast("long"))
        .withColumn("tag_id", F.col("tag_id").cast("long"))
        .withColumn("tag_count", F.col("tag_count").cast("int"))
    )

    print("Loading tag names...", flush=True)
    tags = load_table(spark, "tag", [(0, "tag_id"), (1, "tag")]).withColumn(
        "tag_id", F.col("tag_id").cast("long")
    )

    print("Loading recognized genres...", flush=True)
    genres = (
        load_table(spark, "genre", [(2, "genre_name")])
        .select(F.lower(F.trim("genre_name")).alias("genre_name"))
        .distinct()
    )

    print("Loading trained artist IDs...", flush=True)
    model_artists = spark.read.parquet(str(ARTIST_IDS)).select("artist_mbid").distinct()

    print("Joining metadata...", flush=True)
    # Restrict to artists represented in our ALS model.
    result = (
        artists.join(model_artists, "artist_mbid", "inner")
        .join(artist_tags, "mb_artist_id", "inner")
        .join(tags, "tag_id", "inner")
        .withColumn("tag", F.lower(F.trim("tag")))
        .join(F.broadcast(genres), F.col("tag") == F.col("genre_name"), "left")
        .withColumn("is_genre", F.col("genre_name").isNotNull())
        .select("artist_mbid", "artist_name", "tag", "tag_count", "is_genre")
    )

    print("Saving artist metadata...", flush=True)
    result.write.mode("overwrite").parquet(str(ARTIST_TAGS))

    print("Computing coverage...", flush=True)
    saved = spark.read.parquet(str(ARTIST_TAGS))
    total_artists = model_artists.count()
    tagged_artists = saved.select("artist_mbid").distinct().count()
    genre_artists = saved.filter("is_genre").select("artist_mbid").distinct().count()
    print(f"Trained artists: {total_artists:,}")
    print(f"Artists with tags: {tagged_artists:,}")
    print(f"Tag coverage: {tagged_artists / total_artists:.1%}")
    print(f"Artists with genres: {genre_artists:,}")
    print(f"Genre coverage: {genre_artists / total_artists:.1%}")

    print("\nMost common tags:")
    (
        saved.groupBy("tag")
        .agg(F.countDistinct("artist_mbid").alias("artists"))
        .orderBy(F.desc("artists"))
        .show(30, truncate=False)
    )
    print("\nDone.", flush=True)

    spark.stop()


if __name__ == "__main__":
    main()

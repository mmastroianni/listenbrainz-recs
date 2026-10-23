"""Local Spark session setup shared by the pipeline and experiment scripts."""

from pyspark.sql import SparkSession


def get_spark(
    app_name: str,
    *,
    master: str = "local[4]",
    shuffle_partitions: int = 32,
    log_level: str = "WARN",
) -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel(log_level)
    return spark

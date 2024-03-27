from pyspark.sql import *

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("HandwritingRecognitionModel") \
        .getOrCreate()

    sc = spark.sparkContext
    print(sc)

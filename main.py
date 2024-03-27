from pyspark.sql import *

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("HandwritingRecognitionModel") \
        .getOrCreate()

    sc = spark.sparkContext
    print(sc)

    df = (spark.read.option("header", "")
          .csv("./data/emnist-balanced-train.csv", header=False))
    print(df.count())

    df.show()
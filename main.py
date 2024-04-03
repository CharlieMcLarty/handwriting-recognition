import pyspark
from pyspark.sql import *
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql.types import FloatType

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("HandwritingRecognitionModel") \
        .master("local[*]") \
        .getOrCreate()

    # Balanced dataset has equal amount of samples for each character
    # First column is the label, other 784 represent each pixel for 28x28 image
    train_df = (spark.read
                .option("inferSchema", "true")
                .csv("./data/emnist-balanced-train.csv", header=False))
    train_df = train_df.withColumnRenamed("_c0", "label")
    mapping = spark.read.text("./data/emnist-balanced-mapping.txt")

    # Create separate dataframe for labels and pixels
    labels = train_df.select("label")
    pixels = train_df.drop("label")

    # Change column types to float from string (takes long time to run)
    # for col in pixels.columns:
    #     pixels = pixels.withColumn(col, pixels[col].cast(FloatType()))

    # pixels.printSchema()

    # Prints shape of labels and pixels dataframe
    print((labels.count(), len(labels.columns)))
    print((pixels.count(), len(pixels.columns)))

    print(pixels.dtypes)

    # plt.imshow(pixels.take(1)[0])
    # plt.show()

import pyspark
from pyspark.sql import *
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("HandwritingRecognitionModel") \
        .master("local[*]") \
        .getOrCreate()

    # Balanced dataset has equal amount of samples for each character
    # First column is the label, other 784 represent each pixel for 28x28 image
    train_df = (spark.read
                .csv("./data/emnist-balanced-train.csv", header=False))
    train_df = train_df.withColumnRenamed("_c0", "label")

    labels = train_df.select("label")
    pixels = train_df.drop("label")


# In[1]:
import pyspark
from pyspark.sql import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# In[2]:
# Create Spark session using all available cores
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

# In[3]:
# Create separate dataframe for labels and pixels
labels = train_df.select("label")
pixels = train_df.drop("label")

# In[4]:
# Prints shape of labels and pixels dataframe
print((labels.count(), len(labels.columns)))
print((pixels.count(), len(pixels.columns)))

# Converts first row of pixels to dictionary and extracts the values
# to convert to a 28x28 numpy array
first_row = pixels.take(1)[0].asDict()
pixels_values = list(first_row.values())
print(pixels_values)
numpy_array = np.array(pixels_values, dtype=float).reshape(28, 28)
#
plt.imshow(numpy_array, cmap="gray")
plt.show()

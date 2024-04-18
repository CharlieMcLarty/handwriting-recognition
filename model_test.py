# In[1]:
import os
import keras.models
import numpy as np
from PIL import ImageOps, Image
from keras.preprocessing.image import *
from matplotlib import pyplot as plt

loaded_model = keras.models.load_model('./models/model_v3.keras', compile=False)


# In[2]:
def process_image(img):
    img = load_img(img, color_mode="grayscale", target_size=(28, 28), interpolation="lanczos",
                   keep_aspect_ratio="true")
    img = ImageOps.invert(img)
    plt.imshow(img)
    plt.show()
    img_array = img_to_array(img)
    img_array = np.array(img_array).astype('float32') / 255
    img_array = img_array.reshape((1, 28, 28, 1))
    return img_array

def process(images):
    arr_images = []
    for img in images:
        processed_img = process_image(img)
        arr_images.append(processed_img)
    return np.array(arr_images).reshape(-1,28,28,1)

# In[3]:
def create_dic():
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    mapping_data = spark.sparkContext.textFile("./data/emnist-balanced-mapping.txt").collect()
    mapping = dict(map(lambda x: (int(x.split()[0]), chr(int(x.split()[1]))), mapping_data))
    spark.stop()
    return mapping


def parse_output(prediction, dic):
    return dic[prediction.argmax()%47]


dic = create_dic()
# In[4]:
# Path to your image
img_dir = ["./images/%s"%img for img in os.listdir("./images") if img.endswith("png")]
processed_imgs = process(img_dir)

predictions = loaded_model.predict(processed_imgs)
for prediction in predictions:
    print(parse_output(prediction, dic))

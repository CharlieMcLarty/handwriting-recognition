# In[1]:
import keras.models
import numpy as np
from PIL import ImageOps, Image
from keras.preprocessing.image import *
from matplotlib import pyplot as plt

loaded_model = keras.models.load_model('./models/model_v1.keras', compile=False)


# In[2]:
def process_image(path):
    img = load_img(path, color_mode="grayscale")
    img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.LANCZOS)
    plt.imshow(img)
    plt.show()
    img_array = img_to_array(img)
    img_array = np.array(img_array).astype('float32') / 255
    img_array = img_array.reshape((-1, 28, 28, 1))
    return img_array


# In[3]:
def create_dic():
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    mapping_data = spark.sparkContext.textFile("./data/emnist-balanced-mapping.txt").collect()
    mapping = dict(map(lambda x: (int(x.split()[0]), chr(int(x.split()[1]))), mapping_data))
    spark.stop()
    return mapping


def parse_output(prediction, dic):
    return dic[prediction.argmax()]


dic = create_dic()
# In[4]:
# Path to your image
test_image = "./images/C.png"
predictions = loaded_model.predict(process_image(test_image))
print(parse_output(predictions, dic))

# In[1]:
import keras.models
import tensorflow as tf
from keras.preprocessing.image import *

loaded_model = tf.keras.models.load_model('./models/model_v1.keras')

# In[2]:
def process_image(path):
    img = load_img(path)
    image = img.resize([28, 28])
    img_array = img_to_array(image)
    print(img_array[0])

# In[3]:
# Path to your image
test_image = "./images/E.png"
process_image(test_image)

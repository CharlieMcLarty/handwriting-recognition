# In[1]:
import keras.models
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import *
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from keras.callbacks import EarlyStopping

# In[2]:
# Create Spark session using all available cores
spark = SparkSession.builder \
    .appName("HandwritingRecognitionModel") \
    .config("spark.driver.memory", "8g") \
    .master("local[*]") \
    .getOrCreate()

# In[3]
# Balanced dataset has equal amount of samples for each character
# First column is the label, other 784 represent each pixel for 28x28 image
train_df = (spark.read
            .option("inferSchema", "true")
            .csv("./data/emnist-balanced-train.csv", header=False))
train_df = train_df.withColumnRenamed("_c0", "label")

test_df = (spark.read
           .option("inferSchema", "true")
           .csv("./data/emnist-balanced-test.csv", header=False))
test_df = test_df.withColumnRenamed("_c0", "label")

# print(train_df.count())
#print("No of columns: ", len(train_df.columns), train_df.columns)

# In[4]
# Condenses the pixels into a vector
pixel_cols = ["_c" + str(i + 1) for i in range(784)]
vectorAssembler = VectorAssembler(inputCols=pixel_cols, outputCol="pixels")
train_df = (vectorAssembler
            .transform(train_df)
            .select("label", "pixels")
            .toDF("label", "pixels"))

test_df = (vectorAssembler
           .transform(test_df)
           .select("label", "pixels")
           .toDF("label", "pixels"))

# In[5]:
x_train = np.array(train_df.select("pixels").collect())
x_test = np.array(test_df.select("pixels").collect())


# In[19]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape((-1, 28, 28, 1)).swapaxes(1, 2)
x_test = x_test.reshape((-1, 28, 28, 1)).swapaxes(1, 2)

train_labels = np.array(train_df.select("label").collect())
test_labels = np.array(test_df.select("label").collect())
y_train = tf.keras.utils.to_categorical(train_labels, 47)
y_test = tf.keras.utils.to_categorical(test_labels, 47)

# In[16]:
mapping_data = spark.sparkContext.textFile("./data/emnist-balanced-mapping.txt").collect()
mapping = dict(map(lambda x: (int(x.split()[0]), chr(int(x.split()[1]))), mapping_data))

nrows, ncols = 4, 5
total_images = nrows * ncols

# Select a subset of images and labels from x_train and y_train
indices = np.random.randint(0, x_train.shape[0], total_images)
images = x_train[indices]
labels = train_labels[indices]
labels = labels.flatten()

# Create a figure and arrange the subplots in a grid
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8), squeeze=False)

# Plot each image in the grid along with its label
for i, ax in enumerate(axes.flat):
    if i < total_images:
        image = images[i]
        label = labels[i]
        label_name = mapping[label]
        ax.imshow(image, cmap="gray")
        ax.set_title(label_name)
        ax.axis('off')
    else:
        ax.axis('off')

plt.subplots_adjust(wspace=0.1, hspace=0.3)  # Increased hspace for better title visibility
plt.show()

# In[6]
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(47, activation='softmax')
])

# model.summary()

# In[7]:
model.compile(optimizer="Adam",
              loss="categorical_crossentropy",
              metrics=(["accuracy"]))

# In[8]:
# Prevents overfitting
early_stopping_callback = EarlyStopping(monitor='val_accuracy',
                                        min_delta=0,
                                        verbose=0,
                                        restore_best_weights=True,
                                        patience=3,
                                        mode='max')


# In[9]:
epochs = 10
batch_size = 100

history = model.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_test, y_test),
                    verbose=1,
                    callbacks=[early_stopping_callback])

if early_stopping_callback.stopped_epoch > 0:
    print(
        f"Training was stopped early at epoch {early_stopping_callback.stopped_epoch + 1}")
else:
    print("Training fully completed")

# In[10]:
# Retrieve a list of list results on training and test data
# sets for each training epoch
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))  # Get number of epochs

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.tight_layout()
plt.show()

# Plot training and validation loss per epoch
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.tight_layout()
plt.show()

# In[11]:
model.save("./models/model_v2.keras")

# In[12]:
model2 = keras.models.load_model("./models/model_v2.keras", compile=False)

# In[22]:
y_pred = model2.predict(x_test)
num_samples = 24

# Get random indices for the samples
random_samples = random.sample(range(len(x_test)), num_samples)

# Create a figure with subplots
fig = plt.figure(figsize=(12, 8))

for i, idx in enumerate(random_samples):
    # Create a subplot for each sample
    ax = fig.add_subplot(4, 6, i+1)

    # Display the image
    ax.imshow(x_test[idx], cmap="gray")

    # Get the predicted and true labels
    predicted_label = mapping[y_pred[idx].argmax()]
    true_label = mapping[y_test[idx].argmax()]

    # Set the title with prediction and true label
    ax.set_title("Predicted: {}\nTrue: {}".format(predicted_label, true_label))
    ax.axis("off")

plt.tight_layout()
plt.show()

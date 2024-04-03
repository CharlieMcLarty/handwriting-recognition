# In[1]:
from keras.datasets import mnist
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping

# In[2]:
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

x_train = x_train.reshape((-1,28,28,1))
x_test = x_test.reshape((-1,28,28,1))
y_train = tf.keras.utils.to_categorical(y_train, 47)
y_test = tf.keras.utils.to_categorical(y_test,47)

# In[3]:
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train ,test_size=0.2,random_state = 42)

# In[4]:
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

model.summary()

# In[5]:
model.compile(optimizer="Adam",
         loss="categorical_crossentropy",
         metrics=(["accuracy"]))

# In[6]:
early_stopping_callback = EarlyStopping(monitor='val_accuracy',
                                        min_delta=0,
                                        verbose=0,
                                        restore_best_weights=True,
                                        patience=3,
                                        mode='max')

# In[7]:
epochs = 10
batch_size = 100

# In[8]:
history = model.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_test, y_test),
                    verbose=1,
                    callbacks=[early_stopping_callback])

if early_stopping_callback.stopped_epoch > 0:
    print(
        f"Training was stopped early at epoch {early_stopping_callback.stopped_epoch + 1} due to reaching the desired accuracy.")
else:
    print("Training completed without early stopping.")

# In[9]:
# Retrieve a list of list results on training and test data
# sets for each training epoch
# -----------------------------------------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))  # Get number of epochs

# ------------------------------------------------
# Plot training and validation accuracy per epoch
# ------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.show()
print("")

# ------------------------------------------------
# Plot training and validation loss per epoch
# ------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.show()




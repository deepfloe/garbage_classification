import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from data.get_labelled_data import get_labelled_data
from data.scaled_input_size import scaled_input_size

rescale_factor = 0.5
num_classes = 6


X, y = get_labelled_data(rescale_factor=rescale_factor)

k,m,l = X.shape
input_shape = (m,l,1)
X = X.reshape(k,m,l,1)
print(X.shape)
X = X/255

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.8)
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(4, 4)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64,kernel_size = (3,3),activation = "relu"),
        layers.MaxPooling2D(pool_size = (4,4)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()
batch_size = 128
epochs = 100

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

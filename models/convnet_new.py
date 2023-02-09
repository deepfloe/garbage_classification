from keras import layers
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.model_selection import train_test_split
scaling = 0.3
num_classes = 6
image_size=(int(scaling*384), int(scaling*512))

dataset = image_dataset_from_directory(
    directory='/home/benjamin/PycharmProjects/garbage_classification/raw_jpg/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=image_size)
dataset_train, dataset_test = keras.utils.split_dataset(
    dataset, left_size=0.8, shuffle=False, seed=None)

model = keras.Sequential(
    [
        keras.Input(shape=image_size+(3,)),
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
epochs = 50

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(dataset_train, batch_size=batch_size, epochs=epochs)
score = model.evaluate(dataset_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
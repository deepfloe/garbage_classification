from keras import layers
from tensorflow import keras
num_classes = 6
def get_convnet(scaling):
    image_size = (int(scaling * 384), int(scaling * 512))
    model = keras.Sequential(
        [
            keras.Input(shape=image_size + (3,)),
            layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model

if __name__ == '__main__':

    model = get_convnet(0.6)
    model.summary()
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from data_loader import load_data

def get_data_augmentation_layer():
    ''':returns data_augmentation with random flip, rotation and zoom as a keras Sequential model'''
    data_augmentation = keras.Sequential([layers.RandomFlip("horizontal"),
                                        layers.RandomRotation(0.2),
                                        layers.RandomZoom(0.2),
    ])
    return data_augmentation

def show_augmented_images():
    '''Plots the result of 9 data augmentation maps of a random test image. '''
    train_dataset, _,_ = load_data()
    data_augmentation = get_data_augmentation_layer()
    for data_batch, _ in train_dataset.take(1):
      fig, ax = plt.subplots(nrows = 3, ncols = 3)
      for k in range(9):
          i = k%3
          j = k//3
          augmented_batch = data_augmentation(data_batch)
          ax[i][j].imshow(augmented_batch[0].numpy().astype("uint8"))
    plt.show()

if __name__ == "__main__":
   show_augmented_images()
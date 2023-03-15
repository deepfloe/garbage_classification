from tensorflow.keras.utils import image_dataset_from_directory
import pathlib


def load_data():
    '''Returns train_dataset, val_dataset, test_dataset as tensorflow datasets.'''
    image_size = (384, 512)
    base_path = pathlib.Path("data/train_val_test_split")
    batch_size = 32
    train_dataset = image_dataset_from_directory(base_path/"train",
                                              image_size = image_size,
                                              batch_size = batch_size)
    val_dataset = image_dataset_from_directory(base_path/"val",
                                              image_size = image_size,
                                              batch_size = batch_size)
    test_dataset = image_dataset_from_directory(base_path/"test",
                                              image_size = image_size,
                                              batch_size = batch_size)

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    load_data()
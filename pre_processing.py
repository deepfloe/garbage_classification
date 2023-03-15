import numpy as np
from models import get_conv_base
import keras
from data_loader import load_data


def vgg_preprocessing_datasets(train_dataset, val_dataset, test_dataset):
    '''Version of vgg_preprocessing but returns tensorflow datasets. More elegant but much slower. Not clear to me why.'''
    def preprocess_input(image, label):
        return keras.applications.vgg16.preprocess_input(image), label

    train_dataset_preprocessed = train_dataset.map(preprocess_input)
    val_dataset_preprocessed = val_dataset.map(preprocess_input)
    test_dataset_preprocessed = test_dataset.map(preprocess_input)

    return train_dataset_preprocessed, val_dataset_preprocessed, test_dataset_preprocessed

def vgg_preprocessing(train_dataset, val_dataset, test_dataset):
    '''Applies the convnet base to all test sets. Efficient, as this is only done once and not in every epoch.
    But it cannot be combined with data augmentation.
    :argument tensorflow datasets train_dataset, val_dataset, test_dataset
    :returns a tuple of six numpy arrays
    '''
    def get_features_and_labels(dataset):
        all_features = []
        all_labels = []
        conv_base = get_conv_base()
        for images, labels in dataset:
            preprocessed_images = keras.applications.vgg16.preprocess_input(images)
            features = conv_base(preprocessed_images)
            all_features.append(features)
            all_labels.append(labels)

        return np.concatenate(all_features), np.concatenate(all_labels)

    train_features, train_labels = get_features_and_labels(train_dataset)
    val_features, val_labels = get_features_and_labels(val_dataset)
    test_features, test_labels = get_features_and_labels(test_dataset)
    return train_features, train_labels, val_features, val_labels, test_features, test_labels

if __name__ == "__main__":
    vgg_preprocessing(*load_data())
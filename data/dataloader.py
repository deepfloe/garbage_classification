from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory


num_classes = 6
def load_generators(scaling, batch_size):
    image_size=(int(scaling*384), int(scaling*512))

    train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(directory='/home/benjamin/PycharmProjects/garbage_classification/raw_jpg/train',target_size=image_size,batch_size=batch_size,class_mode='categorical')

    val_generator = train_datagen.flow_from_directory(directory='/home/benjamin/PycharmProjects/garbage_classification/raw_jpg/val',target_size=image_size,batch_size=batch_size,class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(directory='/home/benjamin/PycharmProjects/garbage_classification/raw_jpg/test',target_size=image_size,batch_size=batch_size,class_mode='categorical')
    return train_generator, val_generator, test_generator

def load_tf_dataset(scaling):
    image_size=(int(scaling*384), int(scaling*512))
    dataset = image_dataset_from_directory(
        directory='/home/benjamin/PycharmProjects/garbage_classification/raw_jpg/',
        labels='inferred',
        label_mode='categorical',
        batch_size=32,
        image_size=image_size)
    dataset_train, dataset_test = keras.utils.split_dataset(
        dataset, left_size=0.8, shuffle=False, seed=None)
import keras
from keras import layers
from data_augmentation import get_data_augmentation_layer

def get_conv_base():
    '''Returns convolutional base for vgg16 model'''
    image_size  = (384,  512)
    conv_base = keras.applications.vgg16.VGG16(
      weights = "imagenet",
      include_top = False,
      input_shape = image_size + (3,)
    )
    return conv_base

def get_top_layers():
    '''Returns top layer (trainable) for vgg model.'''
    inputs = keras.Input(shape = (12,16,512))
    x = layers.Flatten()(inputs)
    x = layers.Dense(128)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(6, activation = "softmax")(x)
    top= keras.Model(inputs = inputs, outputs = outputs)
    return top


def get_convnet_from_scratch():
    '''Returns convnet with 4 layers of convolution and max pooling.'''
    image_size = (384, 512)
    data_augmentation = get_data_augmentation_layer()

    inputs = keras.Input(shape=image_size + (3,))

    x = data_augmentation(inputs)

    x = layers.Rescaling(1. / 255)(x)

    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Flatten()(x)

    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(6, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="convnet_from_scratch")
    print(model.summary())
    return model

def get_vgg_augment():
    ''' Returns full vgg16 model with frozen convolutional base and augmentation layer'''
    image_size = (384, 512)
    conv_base = get_conv_base()
    data_augmentation = get_data_augmentation_layer()
    conv_base.trainable = False
    inputs = keras.Input(shape = image_size+(3,))
    x = data_augmentation(inputs)
    x = keras.applications.vgg16.preprocess_input(x)
    x = conv_base(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(6, activation = "softmax")(x)
    model = keras.Model(inputs = inputs, outputs = outputs, name = "vgg_augment")
    print(model.summary())
    return model
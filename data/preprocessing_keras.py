from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale= 1./255)
train_generator = train_datagen.flow_from_directory('data/',classes=['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'])

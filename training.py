import keras
from models import get_convnet_from_scratch, get_top_layers, get_vgg_augment
from data_loader import load_data
from pre_processing import vgg_preprocessing


def get_callback(name):
  callbacks = [
  keras.callbacks.ModelCheckpoint(
  filepath= name +".keras",
  save_best_only=True,
  monitor="val_loss")
  ]
  return callbacks

def train_convnet_from_scratch(epochs):
  train_dataset, val_dataset, _ = load_data()
  model = get_convnet_from_scratch()
  callbacks = get_callback("convnet_from_scratch")
  model.compile(optimizer="rmsprop", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
  history = model.fit(train_dataset, validation_data = val_dataset, epochs = epochs, callbacks = callbacks)
  return history, model

def train_vgg(epochs):
  train_features, train_labels, val_features, val_labels, _, _ = vgg_preprocessing(*load_data())
  model = get_top_layers()
  callbacks = get_callback("vgg")
  model.compile(optimizer="rmsprop", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
  history = model.fit(train_features, train_labels, validation_data = (val_features, val_labels), epochs = epochs, callbacks = callbacks)
  return history, model

def train_vgg_augment(epochs):
  train_dataset, val_dataset, _ = load_data()
  model = get_vgg_augment()
  callbacks = get_callback("vgg_augment")
  model.compile(optimizer="rmsprop", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
  history = model.fit(train_dataset, validation_data = val_dataset, epochs = epochs, callbacks = callbacks)
  return history, model
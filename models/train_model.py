from models.convnet import get_convnet
from data.dataloader import load_generators
from data.plot_model_history import plot_history
from tensorflow import keras

def train_model(model , scaling, epochs, batch_size = 32, save = False):
    test_size = 2019
    val_size = 251
    train_generator, val_generator, test_generator = load_generators(scaling, batch_size)
    steps_per_epoch = int(test_size //batch_size)
    validation_steps = int(val_size //batch_size)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = model.fit(train_generator, steps_per_epoch=steps_per_epoch, validation_data=val_generator, validation_steps=validation_steps, epochs=epochs, callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=3)])
    test_loss, test_accuracy = model.evaluate(test_generator)
    print('test loss', test_loss)
    print('test accuracy', test_accuracy)
    if save:
        model.save(model.name)
    plot_history(history, title = 'History of {} with scaling factor {}'.format(model.name, scaling))

if __name__ == '__main__':
    scaling = 0.2
    model = get_convnet(scaling)
    train_model(model,scaling, epochs = 5, save = False)
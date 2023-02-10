from models.convnet import get_convnet
from data.dataloader import load_generators
from data.plot_model_history import plot_history

def train_model(model , scaling, epochs, batch_size = 32 ):
    test_size = 2019
    val_size = 251
    train_generator, val_generator, test_generator = load_generators(scaling, batch_size)
    steps_per_epoch = int(test_size //batch_size)
    validation_steps = int(val_size //batch_size)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(train_generator, steps_per_epoch=steps_per_epoch, validation_data=val_generator, validation_steps=validation_steps, epochs=epochs)
    test_loss, test_accuracy = model.evaluate(test_generator)
    print('test loss', test_loss)
    print('test accuracy', test_accuracy)
    model.save('garbage_classification')
    plot_history(model.history)

if __name__ == '__main__':
    scaling = 1
    model = get_convnet(scaling)
    train_model(model,scaling, epochs = 30)
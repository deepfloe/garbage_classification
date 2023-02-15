import matplotlib.pyplot as plt

def plot_history(history, title):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    # makes them share the x-axis, which are the epochs

    axes[0].plot(train_loss, color='blue', label='train_loss')
    axes[0].plot(val_loss, color='orange', label='val_loss')
    axes[0].legend()

    axes[1].plot(train_acc, color='blue', label='train_acc')
    axes[1].plot(val_acc, color='orange', label='val_acc')
    axes[1].set_xlabel('epochs')
    axes[1].legend()

    plt.suptitle(title)
    plt.show()

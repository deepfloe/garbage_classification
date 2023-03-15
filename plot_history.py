import matplotlib.pyplot as plt
def plot_history(history):
  '''Plots loss and accuracy in two subplots for training and validation.'''
  fig, ax = plt.subplots(sharex = True, nrows=2, ncols = 1)
  loss = history.history["loss"]
  val_loss = history.history["val_loss"]
  acc = history.history["accuracy"]
  val_acc = history.history["val_accuracy"]
  ax[0].plot(loss, c = 'orange',marker = '.', linestyle = '', label = 'train_loss')
  ax[0].plot(val_loss, c='blue', marker = '.', linestyle = '', label = 'val_loss')
  ax[0].set_xlabel('epochs')
  ax[0].set_ylabel('loss')
  ax[0].legend()
  ax[1].plot(acc, c = 'orange',marker = '.', linestyle = '', label = 'train_acc')
  ax[1].plot(val_acc, c='blue', marker = '.', linestyle = '', label = 'val_acc')
  ax[1].set_xlabel('epochs')
  ax[1].set_ylabel('accuracy')
  ax[1].legend()
  plt.show()
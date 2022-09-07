import matplotlib.pyplot as plt

def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics
  """
  loss = history.history["loss"]

  epochs = range(len(history.history["loss"]))

  # plot loss
  plt.title("loss")
  plt.xlabel("epochs")
  plt.legend()
  plt.plot(epochs, loss, label="training loss")
  if "val_loss" in history.history.keys():
    val_loss = history.history["val_loss"]
    plt.plot(epochs, val_loss, label="validation loss")

  # plot accuracy
  if "accuracy" in history.history.keys():
    accuracy = history.history["accuracy"]
    plt.figure()
    plt.title("accuracy")
    plt.xlabel("epochs")
    plt.legend()
    plt.plot(epochs, accuracy, label="training accuracy")
    if "val_accuracy" in history.history.keys():
      val_accuracy = history.history["val_accuracy"]
      plt.plot(epochs, val_accuracy, label="validation accuracy")

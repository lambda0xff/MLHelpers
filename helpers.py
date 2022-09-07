import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics
  """
  loss = history.history["loss"]

  epochs = range(len(history.history["loss"]))

  # plot loss
  plt.plot(epochs, loss, label="training loss")
  if "val_loss" in history.history.keys():
    val_loss = history.history["val_loss"]
    plt.plot(epochs, val_loss, label="validation loss")

  plt.title("loss")
  plt.xlabel("epochs")
  plt.legend()

  # plot accuracy
  if "accuracy" in history.history.keys():
    accuracy = history.history["accuracy"]
    plt.figure()
    plt.plot(epochs, accuracy, label="training accuracy")
    if "val_accuracy" in history.history.keys():
      val_accuracy = history.history["val_accuracy"]
      plt.plot(epochs, val_accuracy, label="validation accuracy")

    plt.title("accuracy")
    plt.xlabel("epochs")
    plt.legend()


def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix without normalization")

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

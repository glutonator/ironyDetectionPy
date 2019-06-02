import datetime
import matplotlib.pyplot as plt
from keras.callbacks import History


def generate_plots(results: History, path):
    # time = datetime.datetime.now().time()
    print(results.history.keys())

    saveToFile = "loss"
    plt.plot(results.history['loss'], 'r', linewidth=3.0)
    plt.plot(results.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Val Loss'], fontsize=18)
    plt.xlabel('Number of epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Plot', fontsize=16)
    # plt.title('Loss Plot' + str(time), fontsize=16)
    # plt.show()
    # plt.clf()
    plt.savefig(path + saveToFile)
    plt.clf()

    saveToFile = "accuracy"
    plt.plot(results.history['acc'], 'r', linewidth=3.0)
    plt.plot(results.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Val Accuracy'], fontsize=18)
    plt.xlabel('Number of epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    # plt.title('Accuracy Plot' + str(time), fontsize=16)
    plt.title('Accuracy Plot', fontsize=16)
    # plt.show()
    # plt.clf()
    plt.savefig(path + saveToFile)
    plt.clf()

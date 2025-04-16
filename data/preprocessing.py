import numpy as np
from tensorflow.keras.datasets import cifar10

def preprocess_and_save():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    np.save('data/x_train.npy', x_train)
    np.save('data/y_train.npy', y_train)
    np.save('data/x_test.npy', x_test)
    np.save('data/y_test.npy', y_test)

if __name__ == "__main__":
    preprocess_and_save()
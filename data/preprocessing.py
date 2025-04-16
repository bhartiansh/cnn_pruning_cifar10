from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10
import tensorflow as tf

def load_cifar10_data(test_size=0.2):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Split a validation set from training data
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=test_size, random_state=42, stratify=y_train
    )

    # One-hot encoding (optional)
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_val = tf.keras.utils.to_categorical(y_val, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

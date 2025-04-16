# preprocessing.py
from sklearn.model_selection import train_test_split
import tensorflow as tf

def load_cifar10_data(test_size=0.2):
    (x, y), _ = tf.keras.datasets.cifar10.load_data()
    x = x.astype('float32') / 255.0
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=test_size, stratify=y, random_state=42)
    return (x_train, y_train), (x_val, y_val)

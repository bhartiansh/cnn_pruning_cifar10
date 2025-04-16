# preprocessing/cifar10_preprocessor.py

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def preprocess_cifar10(test_size=0.2):
    """
    Loads and preprocesses the CIFAR-10 dataset, normalizes pixel values, and splits into train/val.
    
    Parameters:
        test_size (float): Fraction of data to use for validation.

    Returns:
        x_train, y_train, x_val, y_val
    """
    (x, y), _ = tf.keras.datasets.cifar10.load_data()
    x = x.astype("float32") / 255.0
    y = y.astype("int")

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=test_size, random_state=42, stratify=y)

    return x_train, y_train, x_val, y_val

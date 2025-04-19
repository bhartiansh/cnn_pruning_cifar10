# train_resnet20.py

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from models.resnet20 import resnet20

def train_model(sparsity=0.5, batch_size=128, epochs=150):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

    # Build and compile the model
    model = resnet20()
    model.build(input_shape=(None, 32, 32, 3))

    # Apply pruning (example, adjust as needed)
    apply_pruning(model, sparsity)

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(x_test, y_test))

def apply_pruning(model, sparsity):
    # Implement your pruning strategy here (e.g., SNIP, One-Cycle, etc.)
    pass
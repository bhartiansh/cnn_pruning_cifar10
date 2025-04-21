import numpy as np
import tensorflow as tf
from models.resnet20 import build_resnet20
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import os

# --------------------------
# Global Magnitude Pruning
# --------------------------
def global_magnitude_pruning(model, sparsity):
    print(f"[INFO] Applying global magnitude pruning with sparsity: {sparsity}")
    all_weights = []
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            weights = layer.get_weights()
            if weights:
                all_weights.append(weights[0].flatten())
    
    all_weights = np.concatenate(all_weights)
    threshold = np.percentile(np.abs(all_weights), sparsity * 100)

    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
            weights = layer.get_weights()
            if weights:
                w, b = weights
                mask = np.abs(w) >= threshold
                layer.set_weights([w * mask, b])

# --------------------------
# Load CIFAR-10 Dataset
# --------------------------
def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

# --------------------------
# Training Function
# --------------------------
def train(model, x_train, y_train, x_test, y_test, epochs, batch_size, checkpoint_path):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                           monitor='val_accuracy',
                                           save_best_only=True,
                                           verbose=1)
    ]

    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_test, y_test),
                        callbacks=callbacks)

    return history

# --------------------------
# Main Training Execution (No Iterations)
# --------------------------
def run_pruning_training(sparsity=0.5, epochs=150, batch_size=128):
    (x_train, y_train), (x_test, y_test) = load_cifar10()


    model = build_resnet20()
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()

    global_magnitude_pruning(model, sparsity)


    train(model, x_train, y_train, x_test, y_test, epochs, batch_size, checkpoint_path)
    print(" Training complete.")



    
    
    

import argparse
import numpy as np
import tensorflow as tf
from models.resnet20 import build_resnet20
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import os

# --------------------------
# Global Pruning Function
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
# Data Preparation
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
# Main Script
# --------------------------
def main(sparsity, epochs, batch_size):
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    model = resnet20()
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()

    # Save initial weights
    initial_weights = model.get_weights()

    # Iterative pruning + retraining loop
    num_iterations = 5
    per_iter_sparsity = sparsity / num_iterations

    for i in range(num_iterations):
        print(f"\n🔁 Iteration {i+1}/{num_iterations} | Sparsity this step: {per_iter_sparsity:.2f}")

        model.set_weights(initial_weights)  # rewind to initial weights
        global_magnitude_pruning(model, per_iter_sparsity * (i + 1))

        checkpoint_path = f"results/lwc_resnet20_sparsity_{sparsity}_iter_{i+1}.h5"
        train(model, x_train, y_train, x_test, y_test, epochs, batch_size, checkpoint_path)

    print("✅ Training complete.")

# --------------------------
# Argparse
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparsity", type=float, default=0.5,
                        help="Total target sparsity (e.g., 0.5 for 50%)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Epochs per pruning iteration")
    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()
    main(args.sparsity, args.epochs, args.batch_size)
    
    
    
    
'''How to run this code in a Jupyter Notebook
# Assuming you have the above code in a file named `train_1_both_weights_connection_resnet20.py`
# You can run the training script from a Jupyter Notebook using the following cells:

# Cell 0: Install required packages
!pip install -q tensorflow-model-optimization
import sys
import os
sys.path.append('../models')  # Adjust this path if necessary
    
# Cell 1: Set your hyperparameters
sparsity = 0.5         # Change this to desired sparsity (e.g., 0.2, 0.5, 0.8)
batch_size = 128       # Batch size (e.g., 64, 128, 256)
epochs = 100           # You can increase this since you're on RTX 4060 now

# Cell 2: Run training script with selected config
!python training/train_1_both_weights_connection_resnet20.py --sparsity {sparsity} --batch_size {batch_size} --epochs {epochs}

'''
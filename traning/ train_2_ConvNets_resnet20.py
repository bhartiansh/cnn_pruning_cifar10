import argparse
import numpy as np
import tensorflow as tf
from models.resnet20 import resnet20
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import os

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

# ------------------------------
# L1-Norm Based Filter Pruning
# ------------------------------
def l1_filter_prune(model, sparsity):
    print(f"[INFO] Applying filter pruning with sparsity {sparsity}")
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            weights = layer.get_weights()
            if not weights:
                continue
            w = weights[0]  # shape: (h, w, in_channels, out_channels)
            b = weights[1] if len(weights) > 1 else None

            # Calculate L1 norm of filters
            l1_norms = np.sum(np.abs(w), axis=(0, 1, 2))  # per out_channel
            num_filters = w.shape[-1]
            num_prune = int(num_filters * sparsity)

            if num_prune == 0:
                continue

            prune_idx = np.argsort(l1_norms)[:num_prune]

            # Zero out pruned filters
            for idx in prune_idx:
                w[:, :, :, idx] = 0.0
                if b is not None:
                    b[idx] = 0.0

            if b is not None:
                layer.set_weights([w, b])
            else:
                layer.set_weights([w])

# ------------------------------
# Training Function
# ------------------------------
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

# ------------------------------
# Main Runner
# ------------------------------
def main(sparsity, epochs, batch_size):
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    model = resnet20()
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()

    l1_filter_prune(model, sparsity)

    checkpoint_path = f"results/l2norm_resnet20_sparsity_{sparsity}.h5"
    train(model, x_train, y_train, x_test, y_test, epochs, batch_size, checkpoint_path)

    print("✅ Training complete.")

# ------------------------------
# CLI Args
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparsity", type=float, default=0.3,
                        help="Fraction of filters to prune (e.g., 0.3 means 30%)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size")

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
sparsity = 0.5         # This controls how many filters to prune (e.g., 0.3 = prune 30%)
batch_size = 128       # Batch size
epochs = 100           # Longer training for better accuracy

# Cell 2: Run training script with selected config
!python training/train_2_ConvNets_resnet20.py --sparsity {sparsity} --batch_size {batch_size} --epochs {epochs}

'''
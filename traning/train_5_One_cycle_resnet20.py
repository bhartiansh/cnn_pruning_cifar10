import argparse
import numpy as np
import tensorflow as tf
from models.resnet20 import resnet20
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler
import os

def one_cycle_lr_schedule(initial_lr, max_lr, cycle_length, step_size):
    def lr_schedule(epoch):
        cycle_epoch = epoch % cycle_length
        if cycle_epoch < step_size:
            return initial_lr + (max_lr - initial_lr) * (cycle_epoch / step_size)
        else:
            return max_lr - (max_lr - initial_lr) * ((cycle_epoch - step_size) / step_size)
    return lr_schedule

def one_cycle_prune(model, sparsity):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            # Pruning logic based on layer weight values
            weight = layer.get_weights()[0]
            abs_weight = np.abs(weight)
            threshold = np.percentile(abs_weight, sparsity * 100)
            pruned_weight = np.where(abs_weight < threshold, 0, weight)
            layer.set_weights([pruned_weight, layer.get_weights()[1]])

def train(args):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

    # Build and compile the model
    model = resnet20()
    model.build(input_shape=(None, 32, 32, 3))

    # Apply One-Cycle pruning before training
    one_cycle_prune(model, args.sparsity)

    # Learning rate schedule for One-Cycle policy
    lr_schedule = one_cycle_lr_schedule(initial_lr=1e-6, max_lr=1e-2, cycle_length=args.epochs, step_size=int(args.epochs / 2))
    lr_scheduler = LearningRateScheduler(lr_schedule)

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train,
              epochs=args.epochs,
              batch_size=args.batch_size,
              validation_data=(x_test, y_test),
              callbacks=[lr_scheduler])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparsity', type=float, default=0.5, help='Pruning sparsity level')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    args = parser.parse_args()

    train(args)
    
    
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
!python training/train_5_One_cycle_resnet20.py --sparsity {sparsity} --batch_size {batch_size} --epochs {epochs}

'''
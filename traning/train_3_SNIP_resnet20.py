import argparse
import numpy as np
import tensorflow as tf
from models.resnet20 import resnet20
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def snip_prune(model, x_batch, y_batch, sparsity):
    with tf.GradientTape() as tape:
        preds = model(x_batch, training=True)
        loss = tf.keras.losses.categorical_crossentropy(y_batch, preds)
    grads = tape.gradient(loss, model.trainable_variables)

    snip_scores = [tf.abs(w * g) for w, g in zip(model.trainable_variables, grads) if g is not None]
    all_scores = tf.concat([tf.reshape(score, [-1]) for score in snip_scores], axis=0)
    k = int((1 - sparsity) * tf.size(all_scores).numpy())
    threshold = tf.sort(all_scores)[k].numpy()

    masks = [(tf.abs(score) > threshold).numpy().astype(np.float32) for score in snip_scores]
    mask_idx = 0
    for i, var in enumerate(model.trainable_variables):
        if 'kernel' in var.name:
            var.assign(var * masks[mask_idx])
            mask_idx += 1

def train(args):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

    model = resnet20()
    model.build(input_shape=(None, 32, 32, 3))

    x_sample = x_train[:args.batch_size]
    y_sample = y_train[:args.batch_size]
    snip_prune(model, x_sample, y_sample, args.sparsity)

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=args.epochs,
              batch_size=args.batch_size,
              validation_data=(x_test, y_test))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparsity', type=float, default=0.5, help='Pruning sparsity level')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=150)
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
import runpy
import sys

sys.argv = [
    'train_4_Dual_Gradient_Based_Iterative_Pruning_resnet20.py',
    '--sparsity', '0.6',
    '--batch_size', '128',
    '--epochs', '150'
]

runpy.run_path('training/train_4_Dual_Gradient_Based_Iterative_Pruning_resnet20.py', run_name="__main__")


'''
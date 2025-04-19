import argparse
import numpy as np
import tensorflow as tf
from models.resnet20 import resnet20
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# -------------------------
# Dual Gradient-Based Rapid Iterative Pruning (Simplified Implementation)
# -------------------------
def dual_gradient_iterative_pruning(model, x_train, y_train, sparsity, iterations=10):
    for i in range(iterations):
        with tf.GradientTape() as tape:
            logits = model(x_train[:256], training=True)
            loss = tf.keras.losses.categorical_crossentropy(y_train[:256], logits, from_logits=True)
        grads = tape.gradient(loss, model.trainable_weights)
        score = [tf.abs(g * w) for g, w in zip(grads, model.trainable_weights)]
        all_scores = tf.concat([tf.reshape(s, [-1]) for s in score], axis=0)
        threshold = tf.sort(all_scores)[int((1 - sparsity) * tf.size(all_scores))]
        masks = [(tf.abs(g * w) > threshold) for g, w in zip(grads, model.trainable_weights)]

        # Apply mask
        for var, mask in zip(model.trainable_weights, masks):
            var.assign(var * tf.cast(mask, tf.float32))
    return model

# -------------------------
# Training Function
# -------------------------
def train_model(sparsity, batch_size, epochs):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    model = resnet20()
    model.build(input_shape=(None, 32, 32, 3))
    
    # Apply pruning
    model = dual_gradient_iterative_pruning(model, x_train, y_train, sparsity)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=epochs,
                        batch_size=batch_size)

    model.save(f'models/pruned_resnet20_dual_iterative_sparsity_{sparsity}.h5')
    return history

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparsity', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=120)
    args = parser.parse_args()

    train_model(args.sparsity, args.batch_size, args.epochs)


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
!python training/train_4_Dual_Iterative_Pruning_resnet20.py --sparsity {sparsity} --batch_size {batch_size} --epochs {epochs}

'''
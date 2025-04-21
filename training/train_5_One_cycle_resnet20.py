import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler
from models.resnet20 import build_resnet20  # Make sure this path is correct

import os

# Create model directory if not exists
os.makedirs("models", exist_ok=True)

# -------------------------
# One-Cycle Learning Rate Scheduler
# -------------------------
def one_cycle_lr_schedule(initial_lr, max_lr, cycle_length, step_size):
    def lr_schedule(epoch):
        cycle_epoch = epoch % cycle_length
        if cycle_epoch < step_size:
            return initial_lr + (max_lr - initial_lr) * (cycle_epoch / step_size)
        else:
            return max_lr - (max_lr - initial_lr) * ((cycle_epoch - step_size) / step_size)
    return lr_schedule

# -------------------------
# One-Cycle Pruning (before training)
# -------------------------
def one_cycle_prune(model, sparsity):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            weights = layer.get_weights()
            if len(weights) < 2:
                continue  # skip if bias or weight is missing
            kernel, bias = weights
            abs_weight = np.abs(kernel)
            threshold = np.percentile(abs_weight, sparsity * 100)
            pruned_kernel = np.where(abs_weight < threshold, 0, kernel)
            layer.set_weights([pruned_kernel, bias])
    print(f"One-cycle pruning applied at sparsity {sparsity * 100:.1f}%.")

# -------------------------
# Train the Model
# -------------------------
def train_model(sparsity=0.5, batch_size=128, epochs=150):
    model_path = f"models/pruned_resnet20_onecycle_sparsity_{sparsity}.h5"
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}. Skipping training.")
        return None

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

    model = build_resnet20()
    model.build(input_shape=(None, 32, 32, 3))

    one_cycle_prune(model, sparsity)

    step_size = epochs // 2
    lr_schedule = one_cycle_lr_schedule(initial_lr=1e-6, max_lr=1e-2, cycle_length=epochs, step_size=step_size)
    lr_scheduler = LearningRateScheduler(lr_schedule)

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_test, y_test),
                        callbacks=[lr_scheduler])

    model.save(model_path)
    print(f"Model saved to {model_path}")
    return history
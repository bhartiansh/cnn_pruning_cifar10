import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from models.resnet20 import build_resnet20  # Ensure this is in your Python path

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# -------------------------
# Dual Gradient-Based Iterative Pruning
# -------------------------
def dual_gradient_iterative_pruning(model, x_train, y_train, sparsity, iterations=10):
    for i in range(iterations):
        with tf.GradientTape() as tape:
            logits = model(x_train[:256], training=True)
            loss = tf.keras.losses.categorical_crossentropy(y_train[:256], logits)
        grads = tape.gradient(loss, model.trainable_weights)

        scores = [tf.abs(g * w) for g, w in zip(grads, model.trainable_weights) if g is not None]
        all_scores = tf.concat([tf.reshape(score, [-1]) for score in scores], axis=0)
        k = int((1 - sparsity) * tf.size(all_scores).numpy())
        threshold = tf.sort(all_scores)[k].numpy()

        masks = [(tf.abs(g * w) > threshold) if g is not None else tf.ones_like(w) for g, w in zip(grads, model.trainable_weights)]

        # Apply masks to weights
        for var, mask in zip(model.trainable_weights, masks):
            var.assign(var * tf.cast(mask, tf.float32))

    print(f"✅ Applied Dual Gradient Iterative Pruning at sparsity {sparsity}")
    return model

# -------------------------
# Training Function
# -------------------------
def train_model(sparsity=0.5, batch_size=128, epochs=120):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    model = build_resnet20()
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()

    # Prune the model
    model = dual_gradient_iterative_pruning(model, x_train, y_train, sparsity)

    # Compile and train
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=epochs,
                        batch_size=batch_size)

    # Save model
    model.save(f'models/pruned_resnet20_dual_iterative_sparsity_{sparsity}.h5')
    print("✅ Model training complete and saved.")
    return history
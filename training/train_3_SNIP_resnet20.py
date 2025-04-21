import numpy as np
import tensorflow as tf
from models.resnet20 import build_resnet20
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# -----------------------------
# SNIP Pruning Function
# -----------------------------
def snip_prune(model, x_batch, y_batch, sparsity):
    with tf.GradientTape() as tape:
        preds = model(x_batch, training=True)
        loss = tf.keras.losses.categorical_crossentropy(y_batch, preds)
    grads = tape.gradient(loss, model.trainable_variables)

    snip_scores = [tf.abs(w * g) for w, g in zip(model.trainable_variables, grads) if g is not None and 'kernel' in w.name]
    all_scores = tf.concat([tf.reshape(score, [-1]) for score in snip_scores], axis=0)

    k = int((1 - sparsity) * tf.size(all_scores).numpy())
    threshold = tf.sort(all_scores)[k].numpy()

    masks = [(tf.abs(score) > threshold).numpy().astype(np.float32) for score in snip_scores]
    mask_idx = 0

    for i, var in enumerate(model.trainable_variables):
        if 'kernel' in var.name:
            var.assign(var * masks[mask_idx])
            mask_idx += 1

# -----------------------------
# Training Function
# -----------------------------
def run_snip_training(sparsity=0.5, batch_size=128, epochs=150):
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
    y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

    # Build and initialize model
    model = build_resnet20()
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()

    # SNIP pruning on small batch
    x_sample = x_train[:batch_size]
    y_sample = y_train[:batch_size]
    snip_prune(model, x_sample, y_sample, sparsity)

    # Compile and train
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(x_test, y_test),
              verbose=1)

    print("SNIP pruning and training complete.")
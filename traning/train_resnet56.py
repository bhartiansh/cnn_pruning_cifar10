from models.resnet56_baseline import build_resnet56
from data.cifar10_loader import load_cifar10_data
import tensorflow as tf
import os

def train_model():
    train_gen, val_gen = load_cifar10_data(batch_size=64)
    model = build_resnet56(input_shape=(32, 32, 3), num_classes=10)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'resnet56_cifar10.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1)

    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, restore_best_weights=True)

    history = model.fit(
        train_gen,
        epochs=30,
        validation_data=val_gen,
        callbacks=[checkpoint_cb, earlystop_cb]
    )

    model.save('./models/resnet56_cifar10_final.keras')

if __name__ == "__main__":
    train_model()
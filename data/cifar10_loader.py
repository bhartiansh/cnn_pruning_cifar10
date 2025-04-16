import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_cifar10_data(batch_size=64):
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize
    x_train, x_val = x_train / 255.0, x_val / 255.0

    # Flatten labels
    y_train = y_train.flatten()
    y_val = y_val.flatten()

    # Data augmentation for training
    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(x_train)

    # Training generator
    train_gen = datagen.flow(x_train, y_train, batch_size=batch_size)

    # Use Keras generator for both to keep consistency (optional)
    val_gen = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

    return train_gen, val_gen

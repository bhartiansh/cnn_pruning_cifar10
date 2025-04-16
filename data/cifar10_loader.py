import tensorflow as tf
def load_cifar10_data(batch_size=64):
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()

    x_train, x_val = x_train / 255.0, x_val / 255.0
    y_train = y_train.flatten()
    y_val = y_val.flatten()

    # Convert to tf.data.Dataset and apply augmentations
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(5000).map(lambda x, y: (tf.image.random_flip_left_right(
        tf.image.random_crop(tf.image.resize_with_crop_or_pad(x, 36, 36), [32, 32])), y))
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

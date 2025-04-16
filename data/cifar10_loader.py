def load_cifar10_data(batch_size=64):
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()

    # Normalize
    x_train = x_train.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0

    # Labels must be flat
    y_train = y_train.flatten()
    y_val = y_val.flatten()

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(5000).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

    return train_ds, val_ds

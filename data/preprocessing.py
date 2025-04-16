def load_cifar10_data(test_size=0.2):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=test_size, random_state=42, stratify=y_train
    )

    x_train = x_train.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    return (x_train, y_train), (x_val, y_val)

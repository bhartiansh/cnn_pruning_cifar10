import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from resnet20_baseline import build_resnet20

def lr_schedule(epoch):
    if epoch < 80:
        return 0.1
    elif epoch < 120:
        return 0.01
    else:
        return 0.001

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Normalize with mean and std
mean = x_train.mean(axis=(0,1,2), keepdims=True)
std = x_train.std(axis=(0,1,2), keepdims=True)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Data Augmentation
datagen = ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(x_train)

# Model
model = build_resnet20()
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True),
    metrics=["accuracy"]
)

# Callbacks
checkpoint = ModelCheckpoint("resnet20_cifar10.h5", save_best_only=True, monitor="val_accuracy", mode="max")
lr_scheduler = LearningRateScheduler(lr_schedule)

# Training
model.fit(datagen.flow(x_train, y_train, batch_size=128),
          epochs=200,
          validation_data=(x_test, y_test),
          callbacks=[checkpoint, lr_scheduler])

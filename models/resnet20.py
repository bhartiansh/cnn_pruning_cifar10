# models/resnet20.py
from tensorflow.keras import layers, models

def resnet_block(x, filters, strides=1):
    shortcut = x
    x = layers.Conv2D(filters, 3, strides=strides, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    if strides != 1:
        shortcut = layers.Conv2D(filters, 1, strides=strides)(shortcut)
    x = layers.Add()([x, shortcut])
    return layers.Activation('relu')(x)

def build_resnet20(input_shape=(32, 32, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    for _ in range(3): x = resnet_block(x, 16)
    for _ in range(3): x = resnet_block(x, 32, strides=2 if _ == 0 else 1)
    for _ in range(3): x = resnet_block(x, 64, strides=2 if _ == 0 else 1)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

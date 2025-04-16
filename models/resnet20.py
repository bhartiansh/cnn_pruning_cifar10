# models/resnet20.py
from tensorflow.keras import layers, models
from tensorflow_model_optimization.sparsity import keras as sparsity

def resnet_block(x, filters, strides=1, prune=False):
    Conv2D = sparsity.prune_low_magnitude(layers.Conv2D, **{}) if prune else layers.Conv2D
    
    shortcut = x
    x = Conv2D(filters, 3, strides=strides, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = Conv2D(filters, 3, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)

    if strides != 1:
        shortcut = Conv2D(filters, 1, strides=strides)(shortcut)

    x = layers.Add()([x, shortcut])
    return layers.Activation('relu')(x)

def build_resnet20(input_shape=(32, 32, 3), num_classes=10, prune=False, pruning_schedule=None):
    Conv2D = sparsity.prune_low_magnitude(layers.Conv2D, pruning_schedule=pruning_schedule) if prune else layers.Conv2D

    inputs = layers.Input(shape=input_shape)
    x = Conv2D(16, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    for _ in range(3):
        x = resnet_block(x, 16, prune=prune)
    for i in range(3):
        x = resnet_block(x, 32, strides=2 if i == 0 else 1, prune=prune)
    for i in range(3):
        x = resnet_block(x, 64, strides=2 if i == 0 else 1, prune=prune)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, x)

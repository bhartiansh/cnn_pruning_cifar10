import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import cifar10

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True):
    x = layers.Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    if batch_normalization:
        x = layers.BatchNormalization()(x)
    if activation:
        x = layers.Activation(activation)(x)
    return x

def resnet_block(inputs, num_filters, num_blocks, downsample=False):
    x = inputs
    for i in range(num_blocks):
        strides = 2 if i == 0 and downsample else 1
        y = resnet_layer(x, num_filters, strides=strides)
        y = resnet_layer(y, num_filters, activation=None)

        if i == 0 and downsample:
            x = resnet_layer(x, num_filters, kernel_size=1, strides=2, activation=None, batch_normalization=False)

        x = layers.Add()([x, y])
        x = layers.Activation('relu')(x)
    return x

def build_resnet20(input_shape=(32, 32, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    x = resnet_layer(inputs)

    x = resnet_block(x, 16, num_blocks=3)
    x = resnet_block(x, 32, num_blocks=3, downsample=True)
    x = resnet_block(x, 64, num_blocks=3, downsample=True)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           kernel_initializer='he_normal')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

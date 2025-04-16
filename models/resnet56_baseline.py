from tensorflow.keras import layers, models

def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1,
                 activation='relu', batch_normalization=True, conv_first=True):
    """
    Basic ResNet layer: Conv2D + (optional BN + ReLU) in configurable order.
    """
    x = inputs
    conv = layers.Conv2D(num_filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding='same',
                         kernel_initializer='he_normal')

    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
        x = conv(x)
    return x

def build_resnet56(input_shape=(32, 32, 3), num_classes=10):
    """
    Builds the original ResNet-56 model for CIFAR-10 without any pruning logic.
    """
    num_filters = 16
    num_res_blocks = 9  # 6n + 2 where n=9 → ResNet-56

    inputs = layers.Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)

    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2  # downsample at the start of each stack (except first)

            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)

            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x, num_filters=num_filters,
                                 kernel_size=1, strides=strides, activation=None,
                                 batch_normalization=False)
            x = layers.add([x, y])
            x = layers.Activation('relu')(x)

        num_filters *= 2  # double filters after each stack

    x = layers.AveragePooling2D(pool_size=8)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           kernel_initializer='he_normal')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='ResNet56_CIFAR10_Baseline')
    return model

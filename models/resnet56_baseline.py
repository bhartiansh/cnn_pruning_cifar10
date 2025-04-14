from tensorflow.keras import layers, models

from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude

def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1,
                 activation='relu', batch_normalization=True, conv_first=True,
                 pruning=False, pruning_params=None):
    
    conv_layer = layers.Conv2D(num_filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding='same',
                               kernel_initializer='he_normal')
    
    if pruning and pruning_params:
        conv_layer = prune_low_magnitude(conv_layer, **pruning_params)
    
    x = inputs
    if conv_first:
        x = conv_layer(x)
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation:
            x = layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation:
            x = layers.Activation(activation)(x)
        x = conv_layer(x)

    return x

def build_resnet56(input_shape=(32, 32, 3), num_classes=10):
    num_filters = 16
    num_res_blocks = 9

    inputs = layers.Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)

    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2  # downsample

            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x, num_filters=num_filters,
                                 kernel_size=1, strides=strides, activation=None, batch_normalization=False)
            x = layers.add([x, y])
            x = layers.Activation('relu')(x)
        num_filters *= 2

    x = layers.AveragePooling2D(pool_size=8)(x)
    y = layers.Flatten()(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           kernel_initializer='he_normal')(y)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

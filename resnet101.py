import tensorflow as tf
from tensorflow.keras import layers

"""
Resnet 101
filters - the dimensionality of the output space (i.e. the number of output filters in the convolution).
"""

class BatchNorm(layers.BatchNormalization):
    """
    Note about training values: 
        None: Train BN layers. This is the normal mode
        False: Freeze BN layers. Good when batch size is small
        True: (don't use). Set layer in training mode even when making inferences
    """
    def call(self, input, training=None):
        return super(type(self), self).call(inputs, training = training)

#same dimension 3 layer block
def normal_block(input, filters, use_bias=True, train_bn = True):
    filter1, filter2, filter3 = filters

    x = layers.Conv2D(filters=filter1, kernel_size=(1,1), use_bias=use_bias)(input)
    x = BatchNorm()(x, training=train_bn)
    x = layer.Activation('relu')(x)

    x = layers.Conv2D(filters=filter2, kernel_size=(3,3), padding='same',use_bias=use_bias)(x)
    x = BatchNorm()(x, training=train_bn)
    x = layer.Activation('relu')(x)

    x = layers.Conv2D(filters=filter3, kernel_size=(1,1), use_bias=use_bias)(x)
    x = BatchNorm()(x, training=train_bn)

    x = layers.Add()([x, input])
    x = layers.Activation('relu')(x)

    return x

#dimension halved

def halved_block(input, filters, strides=(2,2), use_bias=True, train_bn = True):
    filter1, filter2, filter3 = filters

    x = layers.Conv2D(filters=filter1, kernel_size=(1,1), use_bias=use_bias)(input)
    x = BatchNorm()(x, training=train_bn)
    x = layer.Activation('relu')(x)

    x = layers.Conv2D(filters=filter2, kernel_size=(3,3), padding='same', use_bias=use_bias)(x)
    x = BatchNorm()(x, training=train_bn)
    x = layer.Activation('relu')(x)

    x = layers.Conv2D(filters=filter3, kernel_size=(1,1), use_bias=use_bias)(x)
    x = BatchNorm()(x, training=train_bn)

    shortcut = layers.Conv2D(filters=filter3, kernel_size=(1,1), strides=strides, use_bias=use_bias)(input)
    shortcut = BatchNorm()(shortcut, training=train_bn)

    x = layers.Add()([x,shortcut])
    x = layers.Activation('relu')(x)

    return x

def build_graph(input, train_bn=True):

    x = layers.ZeroPadding2D((3,3))(input)
    x = layers.Conv2D(64, kernel_size=(7,7), strides=(2,2), use_bias=True)(x)
    x = BatchNorm()(x,training=train_bn)
    x = layers.Activation('relu')(x)
    Stage1 = x = layers.MaxPooling2D((3,3), strides=(2,2), padding="same")(x)

    x = halved_block(x, filters=[64,64,256], strides=(1,1), train_bn=train_bn)
    x = normal_block(x, filters=[64,64,256], train_bn=train_bn)
    Stage2 = x = normal_block(x, filters=[64,64,256], train_bn=train_bn)

    x = halved_block(x, filters=[128,128,512], train_bn=train_bn)
    x = normal_block(x, filters=[128,128,512], train_bn=train_bn)
    x = normal_block(x, filters=[128,128,512], train_bn=train_bn)
    Stage3 = x = normal_block(x, filters=[128,128,512], train_bn=train_bn)

    x = halved_block(x, filters=[256,256,1024], train_bn=train_bn)
    for i in range(21):
        x = normal_block(x, filters=[256,256,1024], train_bn=train_bn)
    Stage4 = x = normal_block(x, filters=[256,256,1024], train_bn=train_bn)

    x = halved_block(x, filters=[512,512,2048], train_bn=train_bn)
    x = normal_block(x, filters=[512,512,2048], train_bn=train_bn)
    Stage5 = x = normal_block(x, filters=[512,512,2048], train_bn=train_bn)

    return [Stage1,Stage2,Stage3,Stage4,Stage5]

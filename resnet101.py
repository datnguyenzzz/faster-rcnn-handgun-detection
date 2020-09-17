import tensorflow as tf
from tensorflow.keras import layers
from utils import BatchNorm

"""
Resnet 101
filters - the dimensionality of the output space (i.e. the number of output filters in the convolution).
"""

#same dimension 3 layer block
def normal_block(input, filters, stage, block, use_bias=True):
    filter1, filter2, filter3 = filters
    conv_name_base = 'res' + str(stage) + block + "_branch"
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filter1, (1,1),name=conv_name_base + '2a', use_bias=use_bias)(input)
    x = BatchNorm(axis=3,name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filter2, (3,3), padding='same', name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=3,name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filter3, (1,1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNorm(axis=3,name=bn_name_base + '2c')(x)

    x = layers.Add()([x, input])
    x = layers.Activation('relu', name='res' + str(stage) + block + '_out')(x)

    return x

#dimension halved

def halved_block(input, filters, stage, block, strides=(2,2), use_bias=True):
    filter1, filter2, filter3 = filters
    conv_name_base = 'res' + str(stage) + block + "_branch"
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filter1, (1,1), strides=strides, name=conv_name_base + '2a', use_bias=use_bias)(input)
    x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filter2, (3,3), padding='same', name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=3,name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filter3, (1,1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filter3, (1,1), strides=strides, name=conv_name_base + '1', use_bias=use_bias)(input)
    shortcut = BatchNorm(axis=3,name=bn_name_base + '1')(shortcut)

    x = layers.Add()([x,shortcut])
    x = layers.Activation('relu', name='res' + str(stage) + block + '_out')(x)

    return x

def build_layers(input):

    x = layers.ZeroPadding2D((3,3))(input)
    x = layers.Conv2D(64, (7,7), strides=(2,2), name='conv1', use_bias=True)(x)
    x = BatchNorm(axis=3, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    Stage1 = x = layers.MaxPooling2D((3,3), strides=(2,2), padding="same")(x)

    x = halved_block(x, filters=[64,64,256], stage=2, block='a', strides=(1,1))
    x = normal_block(x, filters=[64,64,256], stage=2, block='b')
    Stage2 = x = normal_block(x, filters=[64,64,256], stage=2, block='c')

    x = halved_block(x, filters=[128,128,512], stage=3, block='a')
    x = normal_block(x, filters=[128,128,512], stage=3, block='b')
    x = normal_block(x, filters=[128,128,512], stage=3, block='c')
    Stage3 = x = normal_block(x, filters=[128,128,512], stage=3, block='d')

    x = halved_block(x, filters=[256,256,1024], stage=4, block='a')
    for i in range(22):
        x = normal_block(x, filters=[256,256,1024], stage=4, block=chr(98 + i))
    Stage4 = x

    x = halved_block(x, filters=[512,512,2048], stage=5, block='a')
    x = normal_block(x, filters=[512,512,2048], stage=5, block='b')
    Stage5 = x = normal_block(x, filters=[512,512,2048], stage=5, block='c')

    return [Stage1,Stage2,Stage3,Stage4,Stage5]

def build_FPN(C1,C2,C3,C4,C5,config):
    P5 = layers.Conv2D(config.PIRAMID_SIZE, (1,1), name='fpn_c5p5')(C5)

    P4 = layers.Add(name="fpn_p4add")([
        layers.UpSampling2D(size=(2,2), name="fpn_p5upsampled")(P5),
        layers.Conv2D(config.PIRAMID_SIZE, (1,1), name='fpn_c4p4')(C4)
    ])

    P3 = layers.Add(name="fpn_p3add")([
        layers.UpSampling2D(size=(2,2), name="fpn_p4upsampled")(P4),
        layers.Conv2D(config.PIRAMID_SIZE, (1,1), name='fpn_c3p3')(C3)
    ])

    P2 = layers.Add(name="fpn_p2add")([
        layers.UpSampling2D(size=(2,2), name="fpn_p3upsampled")(P3),
        layers.Conv2D(config.PIRAMID_SIZE, (1,1), name='fpn_c2p2')(C2)
    ])

    P2 = layers.Conv2D(config.PIRAMID_SIZE, (3,3), padding="SAME", name="fpn_p2")(P2)
    P3 = layers.Conv2D(config.PIRAMID_SIZE, (3,3), padding="SAME", name="fpn_p3")(P3)
    P4 = layers.Conv2D(config.PIRAMID_SIZE, (3,3), padding="SAME", name="fpn_p4")(P4)
    P5 = layers.Conv2D(config.PIRAMID_SIZE, (3,3), padding="SAME", name="fpn_p5")(P5)
    #P6 for anchor scale in RPN
    P6 = layers.MaxPooling2D(pool_size=(1,1), strides=2, name="fpn_p6")(P5)

    return [P2,P3,P4,P5,P6]

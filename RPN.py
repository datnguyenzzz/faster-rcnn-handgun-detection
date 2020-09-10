import tensorflow as tf
from tensorflow.keras import layers

def build_graph(input_feature, anchor_per_window, anchor_stride):
    shared = layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", strides=anchor_stride)(input_feature)

    #cls
    x = layers.Conv2D(filters=2*anchor_per_window, kernel_size=(1,1), padding="valid", activation="linear")(shared)

    #background/foreground
    rpn_class_ids = layers.Lambda(lambda t : tf.reshape(t, [tf.shape(t)[0],-1,2]))(x)

    #probability background/foregound
    rpn_probs = layers.Activation('softmax')(rpn_class_ids)

    #rgs bounding box offset
    x = layers.Conv2D(filters=4*anchor_per_window, kernel_size=(1,1), padding="valid", activation="linear")(shared)
    rpn_bbox = layers.Lambda(lambda t : tf.reshape(t, [tf.shape(t)[0],-1,4]))(x)

    #return [rpn_class_cls, rpn_probs, rpn_bbox]
    return [rpn_class_ids, rpn_probs, rpn_bbox]

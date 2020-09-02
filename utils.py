import tensorflow as tf
from tensorflow.keras import backend as K

def norm_boxes(boxes, shape):
    h,w = tf.split(K.cast(shape, tf.float32),2)
    scale = K.concatenate((h,w,h,w), axis=-1) - K.constant(1.0)
    shift = K.constant([0.,0.,1.,1.])
    return tf.math.divide(boxes - shift,scale)

def generate_anchors(scales,ratios,anchor_stride,feature_shapes,feature_strides):
    anchors = []
    for i in range(len(scales)):
        scale = scales[i]
        feature_shape = feature_shapes[i]
        feature_stride = feature_strides[i]

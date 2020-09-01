import tensorflow as tf
from tensorflow.keras import backend as K

def norm_boxes(boxes, shape):
    h,w = tf.split(K.cast(shape, tf.float32),2)
    scale = K.concatenate((h,w,h,w), axis=-1) - K.constant(1.0)
    shift = K.constant([0.,0.,1.,1.])
    return tf.math.divide(boxes - shift,scale)

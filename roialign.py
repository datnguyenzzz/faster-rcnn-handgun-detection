import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

import utils


class RoiAlignLayer(layers.Layer):
    def __init__(self,pool_shape,**kwargs):
        super(ProposalLayer,self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self,input):
        rois = input[0]
        image_meta = input[1]
        feature = input[2:]
        #assign feature to each rois
        y1,x1,y2,x2 = tf.split(rois,4,axis=2)
        h = y2 - y1
        w = x2 - x1

        image_shape = utils.parse_image_meta(image_meta)['image_shape'][0]
        spec = utils.log2(tf.square(h*w) / (224.0 / tf.square(tf.cast(image_shape[0] * image_shape[1], tf.float32))))
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(spec), tf.int32)))

    def compute_output_shape(self, input_shape):
        x=1

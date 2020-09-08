import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

import utils


class RoiAlignLayer(layers.Layer):
    def __init__(self,pool_shape,**kwargs):
        super(RoiAlignLayer,self).__init__(**kwargs)
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

        pooled=[]
        roi_to_level=[]
        for i,level in enumerate(range(2,6)):
            ix = tf.where(tf.equal(roi_level, level))
            roi_with_level = tf.gather_nd(rois,ix) #list roi with "level" feature

            roi_ids = tf.cast(ix[:,0], tf.int32)
            roi_to_level.append(ix) #map level with roi list

            #stop gradient propogation
            roi_with_level = tf.stop_gradient(roi_with_level)
            roi_ids = tf.stop_gradient(roi_ids)

            pooled.append(tf.image.crop_and_resize(features[i], roi_with_level, roi_ids, self.pool_shape, method = "bilinear"))

        pooled = tf.concat(pooled, axis=0)

        roi_to_level =  tf.concat(roi_to_level, axis=0)
        roi_range = tf.expand_dims(tf.range(tf.shape(roi_to_level)[0]),1)
        roi_to_level = tf.concat([tf.cast(roi_to_level, tf.int32), roi_range], axis=1)

        ## Rearrange pooled features to match the order of the original boxes
        # Sort roi_to_level by batch then box index
        sorting_ts = roi_to_level[:,0] * 100000 + roi_to_level[:,1]
        ix = tf.nn.top_k(sorting_ts, k = tf.shape(roi_to_level)[0]).indices[::-1]
        ix = tf.gather(roi_to_level[:,2],ix)
        pooled = tf.gather(pooled,ix)

        shape = tf.concat([tf.shape(rois)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )

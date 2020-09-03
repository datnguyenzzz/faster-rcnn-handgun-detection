import numpy as np
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

        scale, ratios = np.meshgrid(np.array(scale), np.array(ratios))
        scale.flatten()
        ratios.flatten()

        h = scale / np.sqrt(ratios)
        w = scale * np.sqrt(ratios)
        """
        The RPN works on the feature map (output of CNN) and defines
        the anchors on the feature map, but the final anchor boxes are
        created with respect to the original image.
        """
        center_y = np.arange(0,feature_shape[0],anchor_stride) * feature_stride
        center_x = np.arange(0,feature_shape[1],anchor_stride) * feature_stride

        center_x,center_y = np.meshgrid(center_x,center_y)
        w,center_x = np.meshgrid(w,center_x)
        h,center_y = np.meshgrid(h,center_y)

        boxes_center = np.stack([center_y,center_x],axis=2).reshape(-1,2)
        boxes_shape = np.stack([h,w],axis=2).reshape(-1,2)

        boxes = np.concatenate([boxes_center - 0.5 * boxes_shape, boxes_center + 0.5 * boxes_shape], axis=1)

        anchors.append(boxes)

    return np.concatenate(anchors, axis=0)

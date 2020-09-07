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

def batch_slice(input, func, batch_size):
    if not isinstance(input,list):
        input = [input]

    output = []
    for i in range(batch_size):
        input_slice = [x[i] for x in inputs]
        output_slice = func(*input_slice)
        if not isinstance(output_slice, (tuple,list)):
            output_slice = [output_slice]
        output.append(output_slice)

    output = list(zip(*output))

    res = [tf.stack(o, axis=0) for o in output]
    if len(res)==1:
        res = res[0]
    return res

def apply_bbox_offset(anchors, bbox_offset):
    """
    anchor = [y1,x1,y2,x2]
    bbox_offset = [dy,dx,log(dh),log(dw)]
    """
    h = anchors[:,2] - anchors[:,0]
    w = anchors[:,3] - anchors[:,1]
    center_y = anchors[:,0] + 0.5 * h
    center_x = anchors[:,1] + 0.5 * w

    center_y += bbox_offset[:,0] * h
    center_x += bbox_offset[:,1] * w
    h *= tf.exp(bbox_offset[:,2])
    w *= tf.exp(bbox_offset[:,3])

    y1 = center_y - 0.5 * h
    x1 = center_x - 0.5 * w
    y2 = y1 + h
    x2 = x1 + w
    res = tf.stack([y1,x1,y2,x2],axis=1)
    return res

def clip_boxes(boxes, window):
    wy1,wx1,wy2,wx2 = tf.split(window,4)
    y1,x1,y2,x2 = tf.split(boxes,4,axis=1)

    y1 = tf.maximum(tf.minimum(y1,wy2),wy1)
    x1 = tf.maximum(tf.minimum(x1,wx2),wx1)
    y2 = tf.maximum(tf.minimum(y2,wy2),wy1)
    x2 = tf.maximum(tf.minimum(x2,wx2),wx1)

    clipped = tf.concat([y1,x1,y2,x2],axis=1)
    clipped.set_shape((clipped.shape[0],4))
    return clipped

def remove_zero_padding(boxes):
    is_zeros = tf.cast(tf.math.reduce_sum(tf.math.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, is_zeros)
    return boxes, is_zeros

#a=tf.constant([[0,0,0,0],[0,1,2,0],[0,3,4,0],[0,0,0,0]])
#print(remove_zero_padding(a))

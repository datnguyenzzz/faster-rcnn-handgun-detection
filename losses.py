import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

import utils

def l1_loss(y_true, y_pred):
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss

def rpn_class_loss_func(input_rpn_match, rpn_class_ids):
    """
    input_rpn_match: [batch, anchor, 1] 1 = positive, -1 negative, 0 neutral
    rpn_class_ids : [batcn,anchoir, 2] BG/FG
    """
    input_rpn_match = tf.squeeze(input_rpn_match, -1)
    #-1/1 => 0/1
    anchor_class = K.cast(K.equal(input_rpn_match, 1), tf.int32)

    id_non_neutral = tf.where(K.not_equal(input_rpn_match,0))
    rpn_class_ids = tf.gather_nd(rpn_class_ids, id_non_neutral)
    anchor_class = tf.gather_nd(anchor_class, id_non_neutral)

    loss = K.sparse_categorical_crossentropy(target=anchor_class, output=rpn_class_ids, from_logits=True)

    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss

def rpn_bbox_loss_func(config, input_rpn_bbox, input_rpn_match, rpn_bbox_offset):
    input_rpn_match = K.squeeze(input_rpn_match, -1)

    id_positive = tf.where(K.equal(input_rpn_match, 1))
    rpn_bbox_offset = tf.gather_nd(rpn_bbox_offset, id_positive)

    batch_counts = K.sum(K.cast(K.equal(input_rpn_match, 1),tf.int32), axis=1)
    input_rpn_bbox = utils.batch_pack(input_rpn_bbox, batch_counts, config.IMAGES_PER_GPU)

    diff = K.abs(input_rpn_bbox - rpn_bbox_offset)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)

    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def rcnn_class_loss_func(target_ids, rcnn_class_ids, total_class_ids):
    target_ids = tf.cast(target_ids, tf.int32)
    rcnn_class_ids = tf.cast(rcnn_class_ids, tf.float32)

    #predictions of classes which are not in dataset (only objects not background)
    pred_class_ids = tf.argmax(rcnn_class_ids, axis=2)
    pred_class = tf.gather(total_class_ids[0], pred_class_ids)
    pred_class = tf.cast(pred_class,tf.float32)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_ids, logits=rcnn_class_ids)
    loss = loss * pred_class

    loss = tf.math.reduce_sum(loss) / tf.math.reduce_sum(pred_class)
    return loss

def rcnn_bbox_loss_func(target_bbox, target_ids, rcnn_bbox):
    #reshape and merge batch and roi_num
    target_ids = K.reshape(target_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1,4))
    rcnn_bbox = K.reshape(rcnn_bbox, (-1,K.int_shape(rcnn_bbox)[2],4))

    #only foreground gt contribute to loss.
    foreground_ix = tf.where(target_ids > 0)[:,0]
    foreground_ids = tf.cast(tf.gather(target_ids,foreground_ix), tf.int64)
    index = tf.stack([foreground_ix, foreground_ids], axis=1)

    target_bbox = tf.gather(target_bbox, foreground_ix)
    rcnn_bbox = tf.gather_nd(rcnn_bbox, index)

    loss = K.switch(tf.size(target_bbox) > 0, l1_loss(target_bbox,rcnn_bbox), tf.constant(0.0))
    loss = K.mean(loss)
    return loss


#a = rcnn_class_loss_func(tf.constant([[1,1,0,0]]), tf.constant([[[0,0],[1,0],[1,1],[0,1]]]),tf.constant([[0,1]]))
#print(tf.math.reduce_mean(a, keepdims=True))

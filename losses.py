import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

def rpn_class_loss_func(input_rpn_match, rpn_class_ids):
    """
    input_rpn_match: [batch, anchor, 1] 1 = positive, -1 negative, 0 neutral
    rpn_class_ids : [batcn,anchoir, 2] BG/FG
    """
    input_rpn_match = tf.squeeze(input_rpn_match, -1)
    #-1/1 => 0/1
    anchor_class = K.cast(k.equal(input_rpn_match, 1), tf.int32)

    id_non_neutral = tf.where(K.not_equal(input_rpn_match,0))
    rpn_class_ids = tf.gather_nd(rpn_class_ids, id_non_neutral)
    anchor_class = tf.gather_nd(anchor_class, id_non_neutral)

    loss = K.sparse_categorical_crossentropy(target=anchor_class, output=rpn_class_ids, from_logits=True)

    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss

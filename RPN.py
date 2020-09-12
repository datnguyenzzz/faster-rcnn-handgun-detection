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

def build_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """
    given anchor and gt boxes, compute overlap, positive anchors and offset to refine them
    """
    #1 = positive anchors, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    #offset
    rpn_bbox = np.zeros((config.self.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    crowd_ids = tf.where(gt_class_ids < 0)[0]
    if crowd_ids.shape[0] > 0:
        non_crowd_ids = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ids]
        gt_class_ids = gt_class_ids[non_crowd_ids]
        gt_boxes = gt_boxes[non_crowd_ids]

        crowd_IoU = utils.IoU_overlap(anchors, crowd_boxes)
        crowd_IoU_max = np.amax(crowd_IoU, axis=1)
        no_crowd_bool = (crowd_IoU_max < 0.001)
    else:
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    

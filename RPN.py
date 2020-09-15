import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

import utils

def build_graph(input_feature, anchor_per_window, anchor_stride):
    shared = layers.Conv2D(512, (3,3), padding="same", activation="relu",
                           strides=anchor_stride, name="rpn_conv_shared")(input_feature)

    #cls
    x = layers.Conv2D(2*anchor_per_window, (1,1), padding="valid",
                      activation="linear",name='rpn_class_raw')(shared)

    #background/foreground
    rpn_class_ids = layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    #probability background/foregound
    rpn_probs = layers.Activation('softmax', name = "rpn_class_xxx")(rpn_class_ids)

    #rgs bounding box offset
    x = layers.Conv2D(4*anchor_per_window, (1,1), padding="valid",
                      activation="linear", name="rpn_bbox_pred")(shared)
    rpn_bbox = layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    #return [rpn_class_cls, rpn_probs, rpn_bbox]
    return [rpn_class_ids, rpn_probs, rpn_bbox]

def build_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """
    given anchor and gt boxes, compute overlap, positive anchors and offset to refine them
    """
    #1 = positive anchors, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    #offset
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    crowd_ids = np.where(gt_class_ids < 0)[0]

    no_crowd_bool = []

    if crowd_ids.shape[0] > 0:
        non_crowd_ids = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ids]
        gt_class_ids = gt_class_ids[non_crowd_ids]
        gt_boxes = gt_boxes[non_crowd_ids]

        crowd_IoU = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_IoU_max = np.amax(crowd_IoU, axis=1)
        no_crowd_bool = (crowd_IoU_max < 0.001)
    else:
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    overlap = utils.compute_overlaps(anchors, gt_boxes)

    anchors_iou_max_id = np.argmax(overlap, axis=1)
    anchors_iou_max = overlap[np.arange(overlap.shape[0]),anchors_iou_max_id] #max iou with each anchor

    rpn_match[(anchors_iou_max < 0.3) & (no_crowd_bool)] = -1

    gt_iou_max_id = np.argwhere(overlap == np.max(overlap, axis=0))[:,0]
    rpn_match[gt_iou_max_id] = 1
    rpn_match[anchors_iou_max >= 0.7] = 1

    #balance + and - anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids]=0

    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE - np.sum(rpn_match==1))
    if extra > 0:
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids]=0

    #for + anchor, compute offset
    ids = np.where(rpn_match == 1)[0]
    ix = 0

    """
    f = open("D:\My_Code\gun_detection\\todo.txt","w")
    f.write("\noverlap\n")
    f.write(str(np.shape(gt_boxes)) + '\n')
    f.write(str(np.shape(anchors)) + '\n')
    for x in overlap:
        f.write(str(x))
    f.write("\nrpn_match\n")
    for x in rpn_match:
        f.write(str(x))
    f.close()
    """
    for i,a in zip(ids, anchors[ids]):
        gt = gt_boxes[anchors_iou_max_id[i]]

        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w

        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w)
        ]

        rpn_bbox[ix] /= config.BBOX_STD_DEV
        ix+=1

    return rpn_match, rpn_bbox

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

import utils

def detection_graph(rois,gt_ids,gt_boxes,config):
    #remove zero padding
    rois,_ = utils.remove_zero_padding(rois)
    gt_boxes,is_zeros = utils.remove_zero_padding(gt_boxes)
    gt_ids = tf.boolean_mask(gt_ids, is_zeros)

    #handle crowd
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ids = tf.where(gt_ids < 0)[:,0]
    non_crowd_ids = tf.where(gt_ids > 0)[:,0]
    crowd_gt_boxes = tf.gather(gt_boxes, crowd_ids)
    gt_ids = tf.gather(gt_ids, non_crowd_ids)
    gt_boxes = tf.gather(gt_boxes,non_crowd_ids)

    #compute IoU  between rois and crowd/non crowd GT boxes
    IoU_non_crowd = utils.IoU_overlap(rois,gt_boxes)
    IoU_crowd = utils.IoU_overlap(rois,crowd_gt_boxes)
    IoU_crowd_max = tf.reduce_max(IoU_crowd,axis=1)

    IoU_non_crowd_max = tf.reduce_max(IoU_non_crowd,axis=1)
    #positive rois
    positive_ids = tf.where(IoU_non_crowd_max >= 0.5)[:,0]
    #negative rois
    negative_ids = tf.where(tf.logical_and(IoU_non_crowd_max < 0.5, IoU_crowd_max < 0.001))[:,0]

    #subsample ROI. Aim for ratio positive/negative = 1/3
    positive_num = int(config.TRAIN_ROIS_PER_IMAGE * config.POSITIVE_ROI_RATIO)
    #positives
    positive_ids = tf.random.shuffle(positive_ids)[:positive_num]
    positive_num = tf.shape(positive_ids)[0]
    #negatives
    #num+ = r+ * all -> all = num+ / r+ num- = (1-r+)*all -> num- = (1-r+) * num+/r+ = 1/r+ * num+ - num+
    negative_ratio = 1.0 / config.POSITIVE_ROI_RATIO
    negative_num = tf.cast(negative_ratio * tf.cast(positive_num, tf.float32), tf.int32) - positive_num
    negative_ids = tf.random.shuffle(negative_ids)[:negative_num]

    positive_rois = tf.gather(rois, positive_ids)
    negative_rois = tf.gather(rois, negative_ids)

    #aplly +rois to GT boxes
    positive_IoU_non_crowd = tf.gather(IoU_non_crowd, positive_ids)
    #find max IoU value roi
    func = tf.cond(
        tf.greater(tf.shape(positive_IoU_non_crowd)[1],0),
        true_fn = lambda: tf.math.argmax(positive_IoU_non_crowd, axis=1),
        false_fn = lambda: tf.cast(tf.constant([]),tf.int64)
    )
    #find gt boxes relate with maximum IoU roi
    roi_gt_boxes = tf.gather(gt_boxes, func)
    roi_gt_ids = tf.gather(gt_ids, func)
    #compute offset from positive rois to it's closest gt bbox
    bbox_offset = utils.compute_bbox_offset(positive_rois, roi_gt_boxes)
    bbox_offset /= config.BBOX_STD_DEV

    rois = tf.concat([positive_rois,negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0],0)

    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N+P), (0, 0)])
    roi_gt_ids = tf.pad(roi_gt_ids, [(0, N+P)])
    bbox_offset = tf.pad(bbox_offset, [(0, N+P), (0, 0)])

    return rois, roi_gt_ids, bbox_offset

class TrainingDetectionLayer(layers.Layer):
    def __init__(self,config,**kwargs):
        super(TrainingDetectionLayer,self).__init__(**kwargs)
        self.config = config

    def call(self,input):
        rois = input[0]
        gt_ids = input[1]
        gt_boxes = input[2]

        output = utils.batch_slice([rois,gt_ids,gt_boxes], lambda x,y,z : detection_graph(x,y,z,self.config), self.config.IMAGES_PER_GPU)

        return output

    def compute_output_shape(self, input_shape):
        return (None,self.num_proposal,4)

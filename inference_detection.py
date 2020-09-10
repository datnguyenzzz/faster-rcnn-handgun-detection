import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

import utils

def refine_detections(rois, probs, offset, window, config):
    """
    Refine classified proposals and filter overlaps and return final
    detections.
    offset: [N, num_classes, (dy, dx, log(dh), log(dw))]
    """
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32) #find class of each ROI
    index = tf.stack([tf.range(probs.shape[0]), class_ids],axis=1)
    class_scores = tf.gather_nd(probs, index)
    offset_specific = tf.gather_nd(offset, index)

    refined_rois = utils.apply_bbox_offset(rois, offset_specific * config.BBOX_STD_DEV)
    refined_rois = utils.clip_boxes(refined_rois, window)

    #maybe ROI class is background
    #filter them out
    keep = tf.where(class_ids > 0)[:,0]
    #also filter out low confidence box
    conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(conf_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]

    #apply per class NMS
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois,   keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_ids))[:,0]

        class_keep = tf.image.non_max_suppression(
            tf.gather(pre_nms_rois,ixs),
            tf.gather(pre_nms_scores,ixs),
            max_output_size=config.DETECTION_MAX_INSTANCES,
            iou_threshold=config.DETECTION_NMS_THRESHOLD
        )

        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids, dtype=tf.int64)

    nms_keep = tf.reshape(nms_keep,[-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:,0])

    #intersec between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]

    #keep top scores
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k = num_keep, sorted=True)[1]
    keep = tf.gather(keep,top_ids)



class InferenceDetectionLayer(layers.Layer):
    """
    Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.
    """
    def __init__(self,config,**kwargs):
        super(InferenceDetectionLayer,self).__init__(**kwargs)
        self.config = config

    def call(self,input):
        rois = input[0]
        rcnn_class = input[1]
        rcnn_bbox = input[2]
        image_meta = input[3]

        image_shape = utils.parse_image_meta(image_meta)['image_shape'][0]
        window = utils.norm_boxes(utils.parse_image_meta(image_meta)['window'], image_shape[:2])

        detections = utils.batch_slice(
            [rois, rcnn_class, rcnn_bbox, window],
            lambda x, y, z, w: refine_detections(x, y, z, w, self.config),
            self.config.IMAGES_PER_GPU
        )

    def compute_output_shape(self, input_shape):

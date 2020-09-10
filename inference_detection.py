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

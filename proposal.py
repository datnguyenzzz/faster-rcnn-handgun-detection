import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import numpy as np

import utils


class ProposalLayer(layers.Layer):
    def __init__(self,num_proposal,nms_threshold,anchors,config=None,**kwargs):
        super(ProposalLayer,self).__init__(**kwargs)
        self.config = config
        self.num_proposal = num_proposal
        self.nms_threshold = nms_threshold
        self.anchors = anchors.astype(np.float32)

    def call(self,input):
        class_probs = input[0][:,:,1] #begin foreground

        bbox_offset = input[1]
        bbox_offset = bbox_offset * np.reshape(self.config.BBOX_STD_DEV, [1,1,4])

        anchors = self.anchors

        pre_nms_limit = min(self.config.PRE_NMS_LIMIT, self.anchors.shape[0])
        ids = tf.nn.top_k(class_probs, pre_nms_limit, sorted=True,
                            name="top_anchors").indices #find k largest probabilities

        #slice to each batch ( images per process)
        class_probs = utils.batch_slice([class_probs, ids], lambda x,y:tf.gather(x, y),
                                        self.config.IMAGES_PER_GPU)
        bbox_offset = utils.batch_slice([bbox_offset, ids], lambda x,y:tf.gather(x, y),
                                        self.config.IMAGES_PER_GPU)
        anchors = utils.batch_slice(ids, lambda x:tf.gather(anchors,x),
                                    self.config.IMAGES_PER_GPU, names=["pre_nms_anchors"])

        #apply bbox to anchor boxes to get better bounding box closer to the closed Foreground object.
        bboxes = utils.batch_slice([anchors,bbox_offset], lambda x,y : utils.apply_bbox_offset(x,y),
                                   self.config.IMAGES_PER_GPU, names=["refined_anchors"])

        #clip to 0..1 range
        h,w = self.config.IMAGE_SHAPE[:2]
        window = np.array([0,0,h,w],dtype=np.float32)
        bboxes = utils.batch_slice(bboxes, lambda x: utils.clip_boxes(x,window),
                                   self.config.IMAGES_PER_GPU, names=["refined_anchors_clipped"])

        #generate proposal by NMS

        normalized_bboxes = bboxes / np.array([[h,w,h,w]])

        def nms(normalized_bboxes,scores):
            ids = tf.image.non_max_suppression(normalized_bboxes, scores, self.num_proposal,
                                               self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(normalized_bboxes,ids)
            padding = tf.maximum(self.num_proposal - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals

        proposals = utils.batch_slice([normalized_bboxes,class_probs], nms, self.config.IMAGES_PER_GPU)

        return proposals

    def compute_output_shape(self, input_shape):
        return (None,self.num_proposal,4)

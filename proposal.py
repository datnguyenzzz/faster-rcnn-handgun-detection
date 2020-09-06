import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

import utils


class ProposalLayer(layers.Layer):
    def __init__(self,num_proposal,nms_threshold,config=None,**kwargs):
        super(ProposalLayer,self).__init__(**kwargs)
        self.config = config
        self.num_proposal = num_proposal
        self.nms_threshold = nms_threshold

    def call(self,input):
        class_probs = input[0][:,:,1] #begin foreground

        bbox_offset = input[1]
        bbox_offset = bbox_offset * np.reshape(self.config.BBOX_STD_DEV, [1,1,4])

        anchors = input[2]

        pre_nms_limit = K.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        ids = tf.math.top_k(input=class_probs, k=pre_nms_limit, sorted=True).indices #find k largest probabilities

        #slice to each batch ( images per process)
        class_probs = utils.batch_slice([class_probs, ids], lambda x,y:tf.gather(params=x, indices=y), self.config.IMAGES_PER_GPU)
        bbox_offset = utils.batch_slice([bbox_offset, ids], lambda x,y:tf.gather(params=x, indices=y), self.config.IMAGES_PER_GPU)
        anchors = utils.batch_slice([anchors, ids], lambda x,y:tf.gather(params=x, indices=y), self.config.IMAGES_PER_GPU)

        #apply bbox to anchor boxes to get better bounding box closer to the closed Foreground object.

        bboxes = utils.batch_slice([anchors,bbox_offset], lambda x,y : utils.apply_bbox_offset(anchors=x,bbox_offset=y), self.config.IMAGES_PER_GPU)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K


class ProposalLayer(layers.Layer):
    def __init__(self,num_proposal,nms_threshold,config=None,**kwargs):
        super(ProposalLayer,self).__init__(**kwargs)
        self.config = config
        self.num_proposal = num_proposal
        self.nms_threshold = nms_threshold

    def call(self,input):
        class_probs = input[0][:,:,1] #begin foreground

        bbox = input[1]
        bbox = bbox * np.reshape(self.config.BBOX_STD_DEV, [1,1,4])

        anchors = input[2]

        pre_nms_limit = K.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        ids = tf.math.top_k(input=class_probs, k=pre_nms_limit, sorted=True).indices #find k largest probabilities

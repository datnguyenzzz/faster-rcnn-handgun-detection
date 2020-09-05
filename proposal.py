import tensorflow as tf
from tensorflow.keras import layers


class ProposalLayer(layers.Layer):
    def __init__(self,num_proposal,nms_threshold,config=None,**kwargs):
        super(ProposalLayer,self).__init__(**kwargs)
        self.config = config
        self.num_proposal = num_proposal
        self.nms_threshold = nms_threshold

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

import utils

def detection_graph(rois,gt_ids,gt_boxes,config):
    

class ProposalLayer(layers.Layer):
    def __init__(self,config,**kwargs):
        super(ProposalLayer,self).__init__(**kwargs)
        self.config = config

    def call(self,input):
        rois = input[0]
        gt_ids = input[1]
        gt_boxes = input[2]

        output = utils.batch_slice([rois,gt_ids,gt_boxes], lambda x,y,z : detection_graph(x,y,z,self.config), self.config.IMAGES_PER_GPU)

        return output

    def compute_output_shape(self, input_shape):
        return (None,self.num_proposal,4)

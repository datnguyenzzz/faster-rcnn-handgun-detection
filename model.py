import tensorflow as tf
import resnet101

class RCNN():
    def __init__(self,mode,config):
        self.mode = mode
        self.config = config #config hyperparameter
        self.model = self.build(mode=mode, config=config)

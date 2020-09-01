import tensorflow as tf
from src import resnet101

class MaskRCNN():
    def __init__(self,mode,config):
        self.mode = mode
        self.config = config #config hyperparameter
        self.model = self.build(mode=mode, config=config)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
#######customize########
import resnet101
import utils

class RCNN():
    def __init__(self,mode,config):
        self.mode = mode
        self.config = config #config hyperparameter
        self.model = self.build(mode=mode, config=config)

    def build(self,mode,config):
        h,w = config.IMAGE_SHAPE[:2]
        print(h,w)

        input_image = keras.Input(shape = [None,None,config.IMAGE_SHAPE[2]])


        if mode == "train":
            #RPN
            input_rpn_match = keras.Input(shape = [None,1], dtype=tf.int32) #match?
            input_rpn_bbox = keras.Input(shape = [None,4], dtype=tf.float32) #bounding box

            #ground truth
            input_gt_ids = keras.Input(shape = [None], dtype=tf.int32) #GT class IDs
            input_gt_boxes = keras.Input(shape = [None,4], dtype=tf.float32)

            #normalize
            gt_boxes = layers.Lambda(lambda x : utils.norm_boxes(x,K.shape(input_image)[1:3]))(input_gt_boxes)
        elif mode == "inference":
            input_anchors = keras.Input(shape = [None,4], dtype=tf.float32)

        #resnet layer
        C1,C2,C3,C4,C5 = resnet101.build_layers(input = input_image, config.TRAIN_BN)
        #FPN
        P2,P3,P4,P5,P6 = resnet101.build_FPN(C1=C1,C2=C2,C3=C3,C4=C4,C5=C5,config=config)

        RPN_feature = [P2,P3,P4,P5,P6]
        RCNN_feature = [P2,P3,P4,P5]

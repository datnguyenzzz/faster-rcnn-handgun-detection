import os
import json
import skimage.io
import skimage.color
import argparse
import numpy as np
import math

import utils

import display_data

from model import RCNN

DATABASE_DIR = os.path.abspath("../database")
ROOT_DIR = os.path.abspath("../database/images/export")
ANN_LINK = os.path.join(ROOT_DIR, "_annotations.coco.json")
#COCO_WEIGHTS = os.path.join(ROOT_DIR,"pretrained\\resnet101_weights_th.h5")
#COCO_WEIGHTS = os.path.join(ROOT_DIR,"pretrained\\resnet101_rcnn.h5")
#COCO_WEIGHTS = os.path.join(ROOT_DIR,"model\\mask_rcnn.h5")
COCO_WEIGHTS = os.path.join(DATABASE_DIR,"model\\mask_rcnn_balloon.h5")

"""
ANNOTATIONS = "/content/drive/My Drive/database/annotations"
IMAGES = "/content/drive/My Drive/database/images/balloon"
COCO_WEIGHTS = "/content/drive/My Drive/database/model/mask_rcnn_balloon.h5"
"""
class Config():
    def __init__(self):

        self.NAME = "gun"

        self.IMAGES_PER_GPU = 1
        self.NUM_CLASSES = 1 + 1
        self.BATCH_SIZE = self.IMAGES_PER_GPU
        self.DETECTION_MIN_CONFIDENCE = 0.7 #
        #self.IMAGE_MIN_DIM = 416
        #self.IMAGE_MAX_DIM = 416
        self.IMAGE_MIN_DIM = 512
        self.IMAGE_MAX_DIM = 512
        self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM,3])

        """
        all image attributes:
        image_id size = 1
        original shape: [h,w,c] size = 3
        after resize shape: [h,w,c]
        window: (y1,x1,y2,x2)
        scale
        active classes ids
        """

        #for FPN layer
        self.TRAIN_BN = False
        self.PIRAMID_SIZE = 256
        self.BACKBONE_STRIDES = [4,8,16,32,64] #C2,C3,C4,C5,C6

        self.BACKBONE_SHAPES = np.array(
            [[int(math.ceil(self.IMAGE_SHAPE[0] / stride)),
              int(math.ceil(self.IMAGE_SHAPE[1] / stride))]
             for stride in self.BACKBONE_STRIDES])

        #for anchors
        self.ANCHOR_SCALES = (32,64,128,256,512)
        #self.ANCHOR_SCALES = (16,32,64,128,256)
        self.ANCHOR_RATIOS = [0.5,1,2]
        self.ANCHOR_STRIDE = 1
        self.RPN_TRAIN_ANCHORS_PER_IMAGE = 256 #

        #for ROI
        self.NUM_ROI_TRAINING = 2000
        self.NUM_ROI_INFERENCE = 1000
        self.NMS_THRESHOLD = 0.7 #Non-Max suppression for choosing ROI
        self.BBOX_STD_DEV =  np.array([0.1, 0.1, 0.2, 0.2]) #standard deviation
        self.PRE_NMS_LIMIT = 6000
        self.TRAIN_ROIS_PER_IMAGE = 300 #
        self.POSITIVE_ROI_RATIO = 0.33

        self.DETECTION_MAX_INSTANCES = 400 #
        self.DETECTION_NMS_THRESHOLD = 0.3
        self.MAX_GT_INSTANCES = 100

        #ROI Pooling
        self.POOL_SIZE = 7

        #FCs layer size
        self.FPN_CLS_FC_LAYERS = 1024

        #image RGB mean
        #self.MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
        #self.MEAN_PIXEL = np.array([108.07914190458807, 99.7199589494769, 93.13341886188812])
        self.MEAN_PIXEL = np.array([147.4,141.5,137.1])

        #learning
        self.LEARNING_RATE = 0.001
        self.LEARNING_MOMENTUM = 0.9
        self.GRADIENT_CLIP_NORM = 5.0
        self.WEIGHT_DECAY = 0.0005
        self.STEPS_PER_EPOCH = 1000 #
        self.VALIDATION_STEPS = 70 #

class Dataset(utils.Dataset):

    def load_attributes(self, subset):

        self.add_class("gun",1,"gun")

        dataset = json.load(open(ANN_LINK))

        file_name = []
        img_ids = []
        for image in dataset['images']:
            file_name.append(image['file_name'])
            img_ids.append(image['id'])

        total_image = len(file_name)

        if subset == "train":
            start,end = 0,int(total_image * 0.9)
        else:
            start,end = int(total_image * 0.9),total_image

        file_name = file_name[start:end]
        img_ids = img_ids[start:end]

        annotations = []

        for i in range(len(file_name)):
            annotations.append([])

        for image in dataset['annotations']:
            image_id = image['image_id']
            if image_id < start or image_id >= end:
                continue

            image_id_scale = image_id - start
            annotations[image_id_scale].append(image['bbox'])
        for i in range(len(file_name)):
            image_name = file_name[i]
            image_path = os.path.join(ROOT_DIR, image_name)
            h,w = 416,416
            image_id = img_ids[i]
            bboxes = annotations[i]


            self.add_image(
                "gun",
                image_id = image_id,
                path = image_path,
                height = h, width = w,
                bboxes = bboxes
            )



def train(model):
    dataset_train = Dataset()
    dataset_train.load_attributes("train")
    dataset_train.prepare()

    dataset_val = Dataset()
    dataset_val.load_attributes("val")
    dataset_val.prepare()


    #LEARNING_RATE = 0.001
    LEARNING_RATE = 0.0001

    model.train(dataset_train, dataset_val, learning_rate=LEARNING_RATE, epochs = 20)

    #display_data.view(dataset_train, model.config, shuffle=True, batch_size=model.config.BATCH_SIZE)


def display(model):
    dataset_train = Dataset()
    dataset_train.load_attributes("train")
    dataset_train.prepare()

    dataset_val = Dataset()
    dataset_val.load_attributes("val")
    dataset_val.prepare()

    display_data.view(dataset_train, model.config, shuffle=True, batch_size=model.config.BATCH_SIZE)

def detect_image(model, image_path=None):
    print("Image detection start ", image_path)
    image = skimage.io.imread(image_path)

    r = model.detect([image], verbose=1)[0]

    print("ROIS: ",r['rois'])
    print("class ids",r['class_ids'])
    print("Scores",r['scores'])

################################################################################
#command: main python main.py train/inference --weights=coco/last --image=link
#tensorboard --logdir=log_dir
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("command")
parser.add_argument("--weights", required = True)
parser.add_argument("--image", required = False)
parser.add_argument("--video", required = False)
args = parser.parse_args()

config = Config()

if args.command == "train":
    model = RCNN(mode = "train", config = config)
else:
    model = RCNN(mode = "inference", config = config)
#load resnet101 pretrained model
if args.weights == "coco":
    weight_path = COCO_WEIGHTS
    print(weight_path)
elif args.weights == "last":
    weight_path = model.find_last()
    print(weight_path)
else:
    weight_path = args.weights

model.load_weights(weight_path, by_name=True)


if args.command == "train":
    train(model)
elif args.command == "inference":
    if args.image!=None:
        #detect_image(model, image_path=args.image)
        x=1

else:
    display(model)

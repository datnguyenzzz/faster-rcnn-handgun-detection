import os
import json
import skimage.io
import skimage.color
import argparse
import numpy as np

from model import RCNN

ROOT_DIR = os.path.abspath("../database")
ANNOTATIONS = os.path.join(ROOT_DIR, "annotations")
IMAGES = os.path.join(ROOT_DIR,"images")
#COCO_WEIGHTS = os.path.join(ROOT_DIR,"pretrained\\resnet101_weights_th.h5")
#COCO_WEIGHTS = os.path.join(ROOT_DIR,"pretrained\\resnet101_rcnn.h5")
COCO_WEIGHTS = os.path.join(ROOT_DIR,"pretrained\\mask_rcnn.h5")

class TrainConfig():
    def __init__(self):

        self.NAME = "gun"

        self.IMAGES_PER_GPU = 2
        self.NUM_CLASSES = 2
        self.BATCH_SIZE = self.IMAGES_PER_GPU
        self.DETECTION_MIN_CONFIDENCE = 0.9
        self.IMAGE_SHAPE = [800,800,3]
        """
        all image attributes:
        image_id size = 1
        original shape: [h,w,c] size = 3
        after resize shape: [h,w,c]
        window: (y1,x1,y2,x2)
        scale
        active classes ids
        """
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

        #for FPN layer
        self.TRAIN_BN = False
        self.PIRAMID_SIZE = 256
        self.BACKBONE_STRIDES = [4,8,16,32,64] #C2,C3,C4,C5,C6

        #for anchors
        self.ANCHOR_SCALES = [32,64,128,256,512]
        self.ANCHOR_RATIOS = [0.5,1,2]
        self.ANCHOR_STRIDE = 1
        self.RPN_TRAIN_ANCHORS_PER_IMAGE = 256

        #for ROI
        self.NUM_ROI_TRAINING = 2000
        self.NUM_ROI_INFERENCE = 1000
        self.NMS_THRESHOLD = 0.7 #Non-Max suppression for choosing ROI
        self.BBOX_STD_DEV = np.array([0.1,0.1,0.2,0.2]) #standard deviation
        self.PRE_NMS_LIMIT = 6000
        self.TRAIN_ROIS_PER_IMAGE = 200
        self.POSITIVE_ROI_RATIO = 0.33
        self.MAX_GT_INSTANCES = 100

        #ROI Pooling
        self.POOL_SIZE = 7

        #FCs layer size
        self.FPN_CLS_FC_LAYERS = 1024

        #image RGB mean
        self.MEAN_PIXEL = np.array([123.7, 116.8, 103.9])


class InferenceConfig():
    def __init__(self):

        self.NAME = "gun"

        self.IMAGES_PER_GPU = 1
        self.NUM_CLASSES = 2
        self.BATCH_SIZE = self.IMAGES_PER_GPU
        self.DETECTION_MIN_CONFIDENCE = 0.9
        self.IMAGE_SHAPE = [800,800,3]
        """
        all image attributes:
        image_id size = 1
        original shape: [h,w,c] size = 3
        after resize shape: [h,w,c]
        window: (y1,x1,y2,x2)
        scale
        active classes ids
        """
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

        self.DETECTION_MAX_INSTANCES = 100
        self.DETECTION_NMS_THRESHOLD = 0.3

class GunDataset():
    def __init__(self):
        self.num_images = 0
        self.image_ids = []
        self.image_attribuites = []

    def add_image(self,dict):
        self.image_attribuites.append(dict)

    def load_attributes(self, subset):
        #subset = train or val
        dataset_dir = os.path.join(IMAGES, subset)

        """
        VIA annotation json format
        filename:
        size:
        regions:
            shape_attributes
                name
                x
                y
                width
                height'
            region_attributes
        """
        annotation_file = os.path.join(ANNOTATIONS,subset + ".json")
        annotations = json.load(open(annotation_file))
        annotations = list(annotations.values())
        #print(annotations)

        for a in annotations:
            if type(a['regions']) is dict:
                rects = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                rects = [r['shape_attributes'] for r in a['regions']]

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height,width = image.shape[:2]
            self.add_image({
                "image_id" : a['filename'],
                "path" : image_path,
                "height" : height,
                "width" : width,
                "rects" : rects
            })

    def load_image(self, image_id):
        image = skimage.io.imread(self.image_attribuites[image_id]['path'])

        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)

        if image.shape[-1] == 4:
            image = image[..., :3]
        return image


def train(model):
    dataset_train = GunDataset()
    dataset_train.load_attributes("train")


    dataset_val = GunDataset()
    dataset_val.load_attributes("val")


    LEARNING_RATE = 0.001
    model.train(dataset_train, dataset_val, learning_rate=LEARNING_RATE, epochs = 30)


################################################################################
#command: main python main.py train/inference --weights=coco/last
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("command")
parser.add_argument("--weights", required = True)
parser.add_argument("--image", required = False)
parser.add_argument("--video", required = False)
args = parser.parse_args()

if args.command == "train":
    config = TrainConfig()
    model = RCNN(mode = "train", config = config)
else:
    config = InferenceConfig()
    model = RCNN(mode = "inference", config = config)
#load resnet101 pretrained model
if args.weights == "coco":
    weight_path = COCO_WEIGHTS
    print(weight_path)
elif args.weights == "last":
    weight_path = model.find_last()

model.load_weights(weight_path, by_name=True)

if args.command == "train":
    train(model)

import os
import json
import skimage.draw
import argparse
import model

ROOT_DIR = os.path.abspath("../database")
ANNOTATIONS = os.path.join(ROOT_DIR, "annotations")
IMAGES = os.path.join(ROOT_DIR,"images")
COCO_WEIGHTS = os.path.join(ROOT_DIR,"pretrained\\resnet101_weights_th.h5")

class TrainConfig():
    def __init__(self):
        self.IMAGES_PER_GPU = 2
        self.NUM_CLASSES = 2
        self.BATCH_SIZE = 100
        self.DETECTION_MIN_CONFIDENCE = 0.9
        self.IMAGE_SHAPE = [800,800,3]

        #for FPN layer
        self.TRAIN_BN = False
        self.PIRAMID_SIZE = 256
        self.BACKBONE_STRIDES = [4,8,16,32,64] #C2,C3,C4,C5,C6

        #for anchors
        self.ANCHOR_SCALES = [32,64,128,256,512]
        self.ANCHOR_RATIOS = [0.5,1,2]
        self.ANCHOR_STRIDE = 1

        #for ROI
        self.NUM_ROI_TRAINING = 2000
        self.NUM_ROI_INFERENCE = 1000
        self.NMS_THRESHOLD = 0.7 #Non-Max suppression for choosing ROI
        self.BBOX_STD_DEV = np.array([0.1,0.1,0.2,0.2]) #standard deviation
        self.PRE_NMS_LIMIT = 6000

class InferenceConfig():
    def __init__(self):
        self.GPU_NUM = 1
        self.IMAGES_PER_GPU = 1

        self.TRAIN_BN = False

class GunDataset():
    def __init__(self):
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


################################################################################
#main
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("command")
parser.add_argument("--image", required = False)
parser.add_argument("--video", required = False)
args = parser.parse_args()

if args.command == "train":
    config = TrainConfig()
    model = model.RCNN(mode = "train", config = config)
else:
    config = InferenceConfig()
    model = model.RCNN(mode = "inference", config = config)

import os
import json
import skimage.io
import skimage.color
import argparse
import numpy as np
import math

from model import RCNN

ROOT_DIR = os.path.abspath("../database")
ANNOTATIONS = os.path.join(ROOT_DIR, "annotations")
IMAGES = os.path.join(ROOT_DIR,"images")
#COCO_WEIGHTS = os.path.join(ROOT_DIR,"pretrained\\resnet101_weights_th.h5")
#COCO_WEIGHTS = os.path.join(ROOT_DIR,"pretrained\\resnet101_rcnn.h5")
COCO_WEIGHTS = os.path.join(ROOT_DIR,"model\\mask_rcnn.h5")

class TrainConfig():
    def __init__(self):

        self.NAME = "gun"

        self.IMAGES_PER_GPU = 1
        self.NUM_CLASSES = 1 + 1
        self.BATCH_SIZE = self.IMAGES_PER_GPU
        self.DETECTION_MIN_CONFIDENCE = 0.9
        self.IMAGE_MIN_DIM = 832
        self.IMAGE_MAX_DIM = 832
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

        self.MINI_MASK_SHAPE = (56, 56)
        self.MASK_POOL_SIZE = 14
        self.MASK_SHAPE = [28, 28]

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
        self.ANCHOR_RATIOS = [0.5,1,2]
        self.ANCHOR_STRIDE = 1
        self.RPN_TRAIN_ANCHORS_PER_IMAGE = 256

        #for ROI
        self.NUM_ROI_TRAINING = 2000
        self.NUM_ROI_INFERENCE = 1000
        self.NMS_THRESHOLD = 0.7 #Non-Max suppression for choosing ROI
        self.BBOX_STD_DEV =  np.array([0.1, 0.1, 0.2, 0.2]) #standard deviation
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

        #learning
        self.LEARNING_RATE = 0.001
        self.LEARNING_MOMENTUM = 0.9
        self.GRADIENT_CLIP_NORM = 5.0
        self.WEIGHT_DECAY = 0.0001
        self.STEPS_PER_EPOCH = 1000
        self.VALIDATION_STEPS = 50


class InferenceConfig():
    def __init__(self):

        self.NAME = "gun"

        self.IMAGES_PER_GPU = 1
        self.NUM_CLASSES = 2
        self.BATCH_SIZE = self.IMAGES_PER_GPU
        self.DETECTION_MIN_CONFIDENCE = 0.9
        self.IMAGE_MIN_DIM = 832
        self.IMAGE_MAX_DIM = 832
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
        self.DETECTION_MAX_INSTANCES = 100
        self.DETECTION_NMS_THRESHOLD = 0.3

class Dataset():
    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, height, width, polygons):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
            "height": height,
            "width": width,
            "polygons" : polygons
        }
        self.image_info.append(image_info)

    def prepare(self, class_map=None):
        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info) / 2
        self._image_ids = np.arange(self.num_images)

        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    @property
    def image_ids(self):
        return self._image_ids

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image

        #image_id = 2 * image_id

        #print(self.image_info[image_id]['path'])

        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]

        mask = np.zeros([info["height"], info["width"],
                        len(info["polygons"])], dtype=np.uint8)

        for i,p in enumerate(info["polygons"]):
            rr,cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr,cc,i] = 1

        class_ids = np.ones([mask.shape[-1]], dtype=np.int32)

        return mask, class_ids

    def load_attributes(self, subset):

        self.add_class("gun",1,"gun")

        dataset_dir = os.path.join(IMAGES, subset)
        ann_link = ANNOTATIONS + "\\" + subset + ".json"
        annotations = json.load(open(ann_link))

        annotations = list(annotations.values())

        annotations = [a for a in annotations if a['regions']]

        for a in annotations:
            polygons = [r['shape_attributes'] for r in a['regions']]

            object_name = [r['region_attributes'] for r in a['regions']]

            obj_name1 = []
            polygons1 = []

            for i,p in enumerate(object_name):
                #print(i,p)
                if p["object"] == "gun":
                    obj_name1.append(p)
                    polygons1.append(polygons[i])


            #print(obj_name1)
            #print(polygons1)

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            h,w = image.shape[:2]

            self.add_image(
                "gun",
                image_id = a['filename'],
                path = image_path,
                height = h, width = w,
                polygons = polygons1
            )


def train(model):
    dataset_train = Dataset()
    dataset_train.load_attributes("train")
    dataset_train.prepare()

    dataset_val = Dataset()
    dataset_val.load_attributes("val")
    dataset_val.prepare()


    #LEARNING_RATE = 0.001
    LEARNING_RATE = 0.0000005
    model.train(dataset_train, dataset_val, learning_rate=LEARNING_RATE, epochs = 30)

def detect_image(model, image_path=None):
    print("Image detection start")
    image = skimage.io.imread(image_path)

    r = model.detect(image, verbose=1)[0]

################################################################################
#command: main python main.py train/inference --weights=coco/last
#tensorboard --logdir=log_dir
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
else:
    if args.image!=None:
        detect_image(model, image_path=args.image)

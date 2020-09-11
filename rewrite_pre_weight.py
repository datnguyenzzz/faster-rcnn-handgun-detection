import h5py
import os

ROOT_DIR = os.path.abspath("../database")
ANNOTATIONS = os.path.join(ROOT_DIR, "annotations")
IMAGES = os.path.join(ROOT_DIR,"images")
#COCO_WEIGHTS = os.path.join(ROOT_DIR,"pretrained\\resnet101_weights_th.h5")
COCO_WEIGHTS = os.path.join(ROOT_DIR,"pretrained\\resnet101_rcnn.h5")

filename = COCO_WEIGHTS

exclude = ["input_gt_masks","mrcnn_mask","mrcnn_mask_bn1","mrcnn_mask_bn2","mrcnn_mask_bn3","mrcnn_mask_bn4",
           "mrcnn_mask_conv1","mrcnn_mask_conv2","mrcnn_mask_conv3","mrcnn_mask_conv4",
           "mrcnn_mask_deconv", "mrcnn_mask_loss", "roi_align_mask"]



with h5py.File(filename,  "a") as f:
    for ex in exclude:
        del f[ex]

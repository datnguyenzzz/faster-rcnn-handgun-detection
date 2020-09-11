import h5py
import os

ROOT_DIR = os.path.abspath("../database")
ANNOTATIONS = os.path.join(ROOT_DIR, "annotations")
IMAGES = os.path.join(ROOT_DIR,"images")
COCO_WEIGHTS = os.path.join(ROOT_DIR,"pretrained\\resnet101_weights_th.h5")

filename = COCO_WEIGHTS


with h5py.File(filename, "r") as f:
    if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']
    print(f)

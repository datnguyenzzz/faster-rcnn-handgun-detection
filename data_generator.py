import tensorflow as tf
import numpy as np
import math
import json

import RPN

import utils

def compose_image_meta(image_id, original_image_shape, image_shape, window, scale, active_class_ids):
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +           # size=3
        list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +                     # size=1
        list(active_class_ids)        # size=num_classes
    )
    return meta

def load_image_gt(dataset, config, image_id):
    image = dataset.load_image(image_id) #
    shape = image.shape

    #for image meta
    window = (0,0,shape[0],shape[1])
    scale = 1
    active_class_ids = np.ones(1, dtype=np.int32) #only have 1 active class

    bboxes = dataset.image_attribuites[image_id]['rects']

    gt_boxes = np.zeros([len(bboxes),4], dtype=np.int32)
    class_ids = np.ones(len(bboxes), dtype=np.int32)#

    ix=0
    for box in bboxes:
        x1,y1 = box['x'],box['y']
        w,h = box['width'], box['height']
        x2,y2 = x1+w,y1+h

        gt_boxes[ix] = np.array([y1,x1,y2,x2])
        ix+=1

    image_meta = compose_image_meta(image_id, shape, shape, window, scale, active_class_ids)

    return image, image_meta, class_ids, gt_boxes

def gen(dataset, config, shuffle=True, random_rois=0, batch_size=1, detection_targets=False):
    """
    shuffle: shuffle image every epoch
    return:
    - input_image
    - input_image_meta
    - input_rpn_match
    - input_rpn_bbox
    - input_gt_class_ids
    - input_gt_boxes
    """
    backbone_shape = np.array([[int(math.ceil(config.IMAGE_SHAPE[0]/stride)),int(math.ceil(config.IMAGE_SHAPE[1]/stride))] for stride in config.BACKBONE_STRIDES])
    anchors = utils.generate_anchors(config.ANCHOR_SCALES,
                                     config.ANCHOR_RATIOS,
                                     config.ANCHOR_STRIDE,
                                     backbone_shape,
                                     config.BACKBONE_STRIDES)

    num_images = len(dataset.image_attribuites)
    image_ids = np.arange(num_images)
    error_count = 0

    index=-1

    while True:
        try:
            index = (index+1) % len(image_ids)

            if shuffle and index==0:
                np.random.shuffle(image_ids)

            image_id = image_ids[index]

            input_image, input_image_meta, input_gt_class_ids, input_gt_boxes = load_image_gt(dataset, config, image_id)

            if not np.any(input_gt_class_ids > 0):
                continue

            #RPN targets
            rpn_match, rpn_bbox = RPN.build_targets(input_image.shape, anchors, input_gt_class_ids, input_gt_boxes, config)


        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            print("Error processing image: ", dataset.image_attribuites[image_id])
            error_count += 1
            if error_count > 5:
                raise

    return None

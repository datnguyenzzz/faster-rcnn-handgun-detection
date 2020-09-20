import tensorflow as tf
import numpy as np
import math
import json
import random

import RPN

import utils

def load_image_gt(dataset, config, image_id, augment=False):
    image_id = int(image_id)
    image = dataset.load_image(image_id) #
    mask, class_ids = dataset.load_mask(image_id)
    shape = image.shape

    image,window, scale, padding = utils.resize_image(
        image,
        min_dim = config.IMAGE_MIN_DIM,
        max_dim = config.IMAGE_MAX_DIM,
        padding = True
    )

    mask = utils.resize_mask(mask, scale, padding)

    #print(image.shape)
    #print(mask.shape)

    if augment:
        if random.randint(0,1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    #_idx = np.sum(mask, axis=(0, 1)) > 0
    #mask = mask[:, :, _idx]
    #class_ids = class_ids[_idx]
    _idx = np.sum(np.where(mask > 0, 1, 0), axis=(0, 1)) >= 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]

    bbox = utils.extract_bboxes(mask)

    mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    #print(mask.shape)

    #for image meta

    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    image_meta = utils.compose_image_meta(image_id, shape, window, active_class_ids)

    return image, image_meta, class_ids, bbox, mask

def gen(dataset, config, shuffle=True, augment=True, batch_size=1):
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
    anchors = utils.generate_anchors(config.ANCHOR_SCALES,
                                     config.ANCHOR_RATIOS,
                                     config.ANCHOR_STRIDE,
                                     config.BACKBONE_SHAPES,
                                     config.BACKBONE_STRIDES)

    b = 0 #batch index
    image_ids = np.copy(dataset.image_ids)
    #print(image_ids)
    error_count = 0

    index=-1

    while True:
        try:
            index = (index+1) % len(image_ids)

            if shuffle and index==0:
                np.random.shuffle(image_ids)

            image_id = image_ids[index]

            input_image, input_image_meta, input_gt_class_ids, input_gt_boxes, input_gt_masks =\
                load_image_gt(dataset, config, image_id, augment=augment)

            if not np.any(input_gt_class_ids > 0):
                continue

            #RPN targets
            rpn_match, rpn_bbox = RPN.build_targets(input_image.shape, anchors, input_gt_class_ids, input_gt_boxes, config)
            if b==0 :
                #initial
                batch_image = np.zeros((batch_size, ) + input_image.shape, dtype = np.float32)
                batch_image_meta = np.zeros((batch_size, ) + input_image_meta.shape, dtype = input_image_meta.dtype)
                batch_gt_class_ids = np.zeros((batch_size, config.MAX_GT_INSTANCES), dtype = np.int32)
                batch_gt_boxes = np.zeros((batch_size, config.MAX_GT_INSTANCES, 4), dtype = np.int32)
                batch_rpn_match = np.zeros([batch_size, anchors.shape[0], 1], dtype = rpn_match.dtype)
                batch_rpn_bbox = np.zeros([batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype = rpn_bbox.dtype)
                batch_gt_masks = np.zeros((batch_size, config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1],
                                           config.MAX_GT_INSTANCES))

            if input_gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(np.arange(input_gt_boxes.shape[0]),
                                       config.MAX_GT_INSTANCES, replace=False)
                input_gt_class_ids = input_gt_class_ids[ids]
                input_gt_boxes = input_gt_boxes[ids]
                input_gt_masks = input_gt_masks[:,:,ids]

            batch_image[b] = utils.mold_image(input_image.astype(np.float32), config)
            batch_image_meta[b] = input_image_meta
            batch_gt_class_ids[b, :input_gt_class_ids.shape[0]] = input_gt_class_ids
            batch_gt_boxes[b, :input_gt_boxes.shape[0]] = input_gt_boxes
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_gt_masks[b, :, :, :input_gt_masks.shape[-1]] = input_gt_masks

            b+=1

            if b>=batch_size:
                """
                inputs = [batch_image, batch_image_meta, batch_rpn_match, batch_gt_boxes,
                          batch_gt_class_ids,batch_gt_boxes, batch_gt_masks]
                outputs = []
                """
                inputs = [batch_image, batch_image_meta, batch_rpn_match, batch_gt_boxes,
                          batch_gt_class_ids,batch_gt_boxes]
                outputs = []


                """
                f = open("D:\My_Code\gun_detection\\todo.txt","w")
                f.write("\n" + image_id + "\n")
                #f.write("\nbatch_image\n")
                #f.write(str(batch_image))
                #f.write("\nbatch_image_meta\n")
                #f.write(str(batch_image_meta))
                f.write("\nbatch_rpn_match\n")
                for x in batch_rpn_match[0]:
                    f.write(str(x))
                #f.write("\nbatch_gt_boxes\n")
                #f.write(str(batch_gt_boxes))
                #f.write("\nbatch_gt_class_ids\n")
                #f.write(str(batch_gt_class_ids))
                f.write("\nbatch_rpn_boxes\n")
                f.write(batch_rpn_bbox)
                f.close()
                """

                yield inputs, outputs

                b=0

        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            print("Error processing image: ", image_id)
            error_count += 1
            if error_count > 5:
                raise

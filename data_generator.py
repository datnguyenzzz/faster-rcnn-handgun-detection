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
    bboxes = dataset.image_attribuites[image_id]['rects']
    scale = 1

    shape = image.shape
    h,w = image.shape[:2]
    window = (0,0,h,w)

    #for image meta

    active_class_ids = np.ones(2, dtype=np.int32) #only have 2 active class 0:BG, 1:gun
    active_class_ids[0]=0


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

    return image, image_meta, class_ids, gt_boxes.astype(np.int32)

def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL

def gen(dataset, config, shuffle=True, batch_size=1):
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
    backbone_shape = np.array([[int(math.ceil(config.IMAGE_SHAPE[0]/stride)),
                                int(math.ceil(config.IMAGE_SHAPE[1]/stride))]
                                for stride in config.BACKBONE_STRIDES])
    anchors = utils.generate_anchors(config.ANCHOR_SCALES,
                                     config.ANCHOR_RATIOS,
                                     config.ANCHOR_STRIDE,
                                     backbone_shape,
                                     config.BACKBONE_STRIDES)

    b = 0 #batch index
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

            ids = np.where(rpn_match==1)
            idx = rpn_bbox[:6]
            print("\n",ids)
            print("\n",idx)

            if b==0 :
                #initial
                batch_image = np.zeros((batch_size, ) + input_image.shape, dtype = np.float32)
                batch_image_meta = np.zeros((batch_size, ) + input_image_meta.shape, dtype = input_image_meta.dtype)
                batch_gt_class_ids = np.zeros((batch_size, config.MAX_GT_INSTANCES), dtype = np.int32)
                batch_gt_boxes = np.zeros((batch_size, config.MAX_GT_INSTANCES, 4), dtype = np.int32)
                batch_rpn_match = np.zeros([batch_size, anchors.shape[0], 1], dtype = rpn_match.dtype)
                batch_rpn_bbox = np.zeros([batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype = rpn_bbox.dtype)

            if input_gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(np.arange(input_gt_boxes.shape[0]),
                                       config.MAX_GT_INSTANCES, replace=False)
                input_gt_class_ids = input_gt_class_ids[ids]
                input_gt_boxes = input_gt_boxes[ids]

            batch_image[b] = mold_image(input_image.astype(np.float32), config)
            batch_image_meta[b] = input_image_meta
            batch_gt_class_ids[b, :input_gt_class_ids.shape[0]] = input_gt_class_ids
            batch_gt_boxes[b, :input_gt_boxes.shape[0]] = input_gt_boxes
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox

            b+=1

            if b>=batch_size:
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
                b=0

                yield inputs, outputs

        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            print("Error processing image: ", dataset.image_attribuites[image_id])
            error_count += 1
            if error_count > 5:
                raise

import scipy.misc
import tensorflow as tf
import numpy as np
import math
import json
import random
from skimage.transform import resize

import RPN

import utils

def load_image_gt(dataset, config, image_id):
    image_id = int(image_id)
    image = dataset.load_image(image_id) #

    old_shape = 416

    print(image)

    image = resize(image,(config.IMAGE_MAX_DIM,config.IMAGE_MAX_DIM))

    shape = image.shape

    image = image * np.full((shape),255.0)

    window = (0,0,shape[0],shape[1])

    #print(window)
    #print(padding)


    bboxes = dataset.load_bboxes(image_id)
    class_ids = np.ones([bboxes.shape[0]], dtype=np.int32)

    for i,bbox in enumerate(bboxes):
        y1,x1,y2,x2 = bbox

        x1 = x1 * 1.0*config.IMAGE_MAX_DIM / old_shape
        x2 = x2 * 1.0*config.IMAGE_MAX_DIM / old_shape
        y1 = y1 * 1.0*config.IMAGE_MAX_DIM / old_shape
        y2 = y2 * 1.0*config.IMAGE_MAX_DIM / old_shape

        bboxes[i] = np.array([y1,x1,y2,x2])

    """
    if random.randint(0,1):
        import imgaug as ia
        import imgaug.augmenters as iaa

        bbs = []
        for bbox in bboxes:
            y1,x1,y2,x2 = bbox
            bbs.append(ia.BoundingBox(x1=x1,y1=y1,x2=x2,y2=y2))

            image_aug, bbs_aug = iaa.Fliplr(1.0)(image=image, bounding_boxes=bbs)

        image = image_aug

        gt_boxes_aug = np.zeros([len(bbs_aug),4], dtype=np.float32)

        for i,bbox in enumerate(bbs_aug):
            #print(bbox.y1,bbox.x1,bbox.y2,bbox.x2)
            y1,x1,y2,x2 = bbox.y1,bbox.x1,bbox.y2,bbox.x2

            gt_boxes_aug[i] = np.array([y1,x1,y2,x2])

        bboxes = gt_boxes_aug
    """

    #print(mask.shape)

    #for image meta

    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    image_meta = utils.compose_image_meta(image_id, shape, window, active_class_ids)

    return image, image_meta, class_ids, bboxes


def view(dataset, config, shuffle=True, augment=True, batch_size=1):
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
    print(len(image_ids))
    error_count = 0

    index=1723

    while True:
        image_id = image_ids[index]

        input_image, input_image_meta, input_gt_class_ids, input_gt_boxes =\
            load_image_gt(dataset, config, image_id)

        print(input_image.shape)
        print(input_image_meta)
        print(input_gt_class_ids)
        print(input_gt_boxes)

        if not np.any(input_gt_class_ids > 0):
            continue

        #RPN targets
        rpn_match, rpn_bbox = RPN.build_targets(input_image.shape, anchors, input_gt_class_ids, input_gt_boxes, config)

        #print(input_gt_boxes)
        for gt_box in input_gt_boxes:
            y1,x1,y2,x2 = gt_box
            print(y1,x1,y2,x2)

            y1 = int(y1)
            x1 = int(x1)
            y2 = int(y2)
            x2 = int(x2)

            for y in range(y1,y2+1):
                input_image[y][x1][0] = 255.0
                input_image[y][x1][1] = 0.0
                input_image[y][x1][2] = 0.0

            for y in range(y1,y2+1):
                input_image[y][x2][0] = 255.0
                input_image[y][x2][1] = 0.0
                input_image[y][x2][2] = 0.0

            for x in range(x1,x2+1):
                input_image[y1][x][0] = 255.0
                input_image[y1][x][1] = 0.0
                input_image[y1][x][2] = 0.0
            for x in range(x1,x2+1):
                input_image[y2][x][0] = 255.0
                input_image[y2][x][1] = 0.0
                input_image[y2][x][2] = 0.0

        for i,anchor in enumerate(anchors):
            #anchor = utils.clip_boxes(anchor, np.array([0,0,832,832]))
            if rpn_match[i]==0:
                continue
            y1,x1,y2,x2 = anchor

            y1 = max(min(y1,512-1), 0)
            x1 = max(min(x1,512-1), 0)
            y2 = max(min(y2,512-1), 0)
            x2 = max(min(x2,512-1), 0)

            y1 = int(y1)
            x1 = int(x1)
            y2 = int(y2)
            x2 = int(x2)
            #print(y1,x1,y2,x2)
            if rpn_match[i]==1:
                for y in range(y1,y2+1):
                    input_image[y][x1][0] = 0.0
                    input_image[y][x1][1] = 255.0
                    input_image[y][x1][2] = 0.0
                    input_image[y][x2][0] = 0.0
                    input_image[y][x2][1] = 255.0
                    input_image[y][x2][2] = 0.0

                for x in range(x1,x2+1):
                    input_image[y1][x][0] = 0.0
                    input_image[y1][x][1] = 255.0
                    input_image[y1][x][2] = 0.0
                    input_image[y2][x][0] = 0.0
                    input_image[y2][x][1] = 255.0
                    input_image[y2][x][2] = 0.0
            else:
                for y in range(y1,y2+1):
                    input_image[y][x1][0] = 0.0
                    input_image[y][x1][1] = 0.0
                    input_image[y][x1][2] = 255.0
                    input_image[y][x2][0] = 0.0
                    input_image[y][x2][1] = 0.0
                    input_image[y][x2][2] = 255.0

                for x in range(x1,x2+1):
                    input_image[y1][x][0] = 0.0
                    input_image[y1][x][1] = 0.0
                    input_image[y1][x][2] = 255.0
                    input_image[y2][x][0] = 0.0
                    input_image[y2][x][1] = 0.0
                    input_image[y2][x][2] = 255.0
        """
        f = open("todo.txt", "w")
        for x in rpn_match:
            f.write(str(x))
        for x in rpn_bbox:
            f.write(str(x))
        f.close()
        """
        from skimage.io import imsave
        imsave('GT_RPN_input_val.png',input_image)

        break


def inspect(image,rois):

    image,window, scale, padding = utils.resize_image(
        image,
        min_dim = 896,
        max_dim = 896,
        padding = True
    )

    for roi in rois:
        y1,x1,y2,x2 = roi

        y1 = max(min(y1,896-1), 0)
        x1 = max(min(x1,896-1), 0)
        y2 = max(min(y2,896-1), 0)
        x2 = max(min(x2,896-1), 0)

        y1 = int(y1)
        x1 = int(x1)
        y2 = int(y2)
        x2 = int(x2)
        #print(y1,x1,y2,x2)
        for y in range(y1,y2+1):
            image[y][x1][0] = 1.0
            image[y][x1][1] = 0.0
            image[y][x1][2] = 0.0
            image[y][x2][0] = 1.0
            image[y][x2][1] = 0.0
            image[y][x2][2] = 0.0

        for x in range(x1,x2+1):
            image[y1][x][0] = 1.0
            image[y1][x][1] = 0.0
            image[y1][x][2] = 0.0
            image[y2][x][0] = 1.0
            image[y2][x][1] = 0.0
            image[y2][x][2] = 0.0

    from skimage.io import imsave
    imsave('foo.png',image)

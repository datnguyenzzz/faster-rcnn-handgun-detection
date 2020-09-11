import tensorflow as tf
import numpy as np

def data_generator(dataset, config, shuffle=True, augment=False, augmentation=None,
                   random_rois=0, batch_size=1, detection_targets=False,
                   no_augmentation_sources=None):
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

    backbone_shape = np.array([[int(math.ceil(image_shape[0]/stride)),int(math.ceil(image_shape[1]/stride))] for stride in self.config.BACKBONE_STRIDES])
    anchors = utils.generate_anchors(self.config.ANCHOR_SCALES,
                                     self.config.ANCHOR_RATIOS,
                                     self.config.ANCHOR_STRIDE,
                                     backbone_shape,
                                     self.config.BACKBONE_STRIDES)

    while True:
        try:
            pass
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            print("Error processing image: ", dataset.image_attribuites[image_id])
            error_count += 1
            if error_count > 5:
                raise

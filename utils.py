import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from skimage.transform import resize
import scipy

class BatchNorm(layers.BatchNormalization):
    """
    Note about training values:
        None: Train BN layers. This is the normal mode
        False: Freeze BN layers. Good when batch size is small
        True: (don't use). Set layer in training mode even when making inferences
    """
    def call(self, input, training=None):
        return super(type(self), self).call(input, training = training)

def generate_anchors(scales,ratios,anchor_stride,feature_shapes,feature_strides):
    anchors = []
    for i in range(len(scales)):
        scale = scales[i]
        feature_shape = feature_shapes[i]
        feature_stride = feature_strides[i]

        scale, ratios = np.meshgrid(np.array(scale), np.array(ratios))
        scale = scale.flatten()
        ratios = ratios.flatten()

        h = scale / np.sqrt(ratios)
        w = scale * np.sqrt(ratios)
        """
        The RPN works on the feature map (output of CNN) and defines
        the anchors on the feature map, but the final anchor boxes are
        created with respect to the original image.
        """
        center_y = np.arange(0,feature_shape[0],anchor_stride) * feature_stride
        center_x = np.arange(0,feature_shape[1],anchor_stride) * feature_stride
        center_x,center_y = np.meshgrid(center_x,center_y)

        w,center_x = np.meshgrid(w,center_x)
        h,center_y = np.meshgrid(h,center_y)

        boxes_center = np.stack([center_y,center_x],axis=2).reshape([-1,2])
        boxes_shape = np.stack([h,w],axis=2).reshape([-1,2])

        boxes = np.concatenate([boxes_center - 0.5 * boxes_shape,
                                boxes_center + 0.5 * boxes_shape], axis=1)

        anchors.append(boxes)

    return np.concatenate(anchors, axis=0)

def batch_slice(input, func, batch_size, names=None):
    if not isinstance(input,list):
        input = [input]

    output = []
    for i in range(batch_size):
        input_slice = [x[i] for x in input]
        output_slice = func(*input_slice)
        if not isinstance(output_slice, (tuple,list)):
            output_slice = [output_slice]
        output.append(output_slice)

    output = list(zip(*output))

    if names is None:
        names = [None] * len(output)

    res = [tf.stack(o, axis=0, name=n) for o,n in zip(output,names)]
    if len(res)==1:
        res = res[0]
    return res

def apply_bbox_offset(anchors, bbox_offset):
    """
    anchor = [y1,x1,y2,x2]
    bbox_offset = [dy,dx,log(dh),log(dw)]
    """
    h = anchors[:,2] - anchors[:,0]
    w = anchors[:,3] - anchors[:,1]
    center_y = anchors[:,0] + 0.5 * h
    center_x = anchors[:,1] + 0.5 * w

    center_y += bbox_offset[:,0] * h
    center_x += bbox_offset[:,1] * w
    h *= tf.exp(bbox_offset[:,2])
    w *= tf.exp(bbox_offset[:,3])

    y1 = center_y - 0.5 * h
    x1 = center_x - 0.5 * w
    y2 = y1 + h
    x2 = x1 + w
    res = tf.stack([y1,x1,y2,x2],axis=1)
    return res

def clip_boxes(boxes, window):
    wy1,wx1,wy2,wx2 = tf.split(window,4)
    y1,x1,y2,x2 = tf.split(boxes,4,axis=1)

    y1 = tf.maximum(tf.minimum(y1,wy2),wy1)
    x1 = tf.maximum(tf.minimum(x1,wx2),wx1)
    y2 = tf.maximum(tf.minimum(y2,wy2),wy1)
    x2 = tf.maximum(tf.minimum(x2,wx2),wx1)

    clipped = tf.concat([y1,x1,y2,x2],axis=1)
    clipped.set_shape((clipped.shape[0],4))
    return clipped

def remove_zero_padding(boxes):
    is_zeros = tf.cast(tf.math.reduce_sum(tf.math.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, is_zeros)
    return boxes, is_zeros

def IoU_overlap(boxes1, boxes2): #for tf
    #tile boxes2 and loop boxes1
    box1 = tf.reshape(tf.tile(tf.expand_dims(boxes1,1), [1,1,tf.shape(boxes2)[0]]), [-1,4])
    box2 = tf.tile(boxes2, [tf.shape(boxes1)[0],1])

    y11,x11,y21,x21 = tf.split(box1,4,axis=1)
    y12,x12,y22,x22 = tf.split(box2,4,axis=1)

    y1 = tf.maximum(y11,y12)
    x1 = tf.maximum(x11,x12)
    y2 = tf.minimum(y21,y22)
    x2 = tf.minimum(x21,x22)

    I = tf.maximum(x2-x1,0) * tf.maximum(y2-y1,0)
    U = (y21-y11)*(x21-x11) + (y22-y12)*(x22-x12) - I

    #print(tf.where(U==0.0))

    IoU = I/U

    IoU = tf.reshape(IoU, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return IoU

def compute_overlaps(boxes1, boxes2):

    #print("fucknp ",boxes1,boxes2)

    def compute_iou(box, boxes, box_area, boxes_area):
        # Calculate intersection areas
        y1 = np.maximum(box[0], boxes[:, 0])
        y2 = np.minimum(box[2], boxes[:, 2])
        x1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[3], boxes[:, 3])
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        union = box_area + boxes_area[:] - intersection[:]
        #assert(np.where(union==0.0).shape[0]==0)
        #print(len(np.where(union==0.0)))
        iou = intersection / union
        return iou

    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def compute_bbox_offset(rois,gt_boxes):
    """
    input form = (y1,x1,y2,x2)
    return offset = (dy,dx,log(hd),log(dw))
    """
    rois = tf.cast(rois,tf.float32)
    gt_boxes = tf.cast(gt_boxes,tf.float32)
    h = rois[:,2] - rois[:,0]
    w = rois[:,3] - rois[:,1]
    center_y = rois[:,0] + 0.5 * h
    center_x = rois[:,1] + 0.5 * w

    gt_h = gt_boxes[:,2] - gt_boxes[:,0]
    gt_w = gt_boxes[:,3] - gt_boxes[:,1]
    gt_center_y = gt_boxes[:,0] + 0.5 * gt_h
    gt_center_x = gt_boxes[:,1] + 0.5 * gt_w

    assert(h!=0)
    assert(w!=0)
    dy = (gt_center_y - center_y) / h
    dx = (gt_center_x - center_x) / w
    dh = tf.math.log(gt_h / h)
    dw = tf.math.log(gt_w / w)

    res = tf.stack([dy,dx,dh,dw], axis = 1)
    return res

def parse_image_meta(meta):
    image_id = meta[:,0]
    image_shape = meta[:,1:4]
    window = meta[:,4:7]
    class_ids = meta[:,7:]
    return {
        "image_id": image_id,
        "image_shape": image_shape,
        "window": window,
        "class_ids": class_ids
    }

def log2(x):
    return tf.math.log(x) / tf.math.log(2.0)

def batch_pack(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def minimize_mask(bbox, mask, mini_shape):
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        m = resize(m.astype(float), mini_shape)
        mini_mask[:, :, i] = np.where(m >= 128, 1, 0)
    return mini_mask

def resize_image(image, min_dim=None, max_dim=None, padding=False):
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)))
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding


def resize_mask(mask, scale, padding):
    h, w = mask.shape[:2]
    mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask

def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL

def compose_image_meta(image_id, image_shape, window, active_class_ids):
    meta = np.array(
        [image_id] +                  # size=1
        list(image_shape) +           # size=3
        list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates                    # size=1
        list(active_class_ids)        # size=num_classes
    )
    return meta

#a=tf.constant([[0,0,0,0],[0,1,2,0],[0,3,4,0],[0,0,0,0]])
#print(tf.math.argmax(a))

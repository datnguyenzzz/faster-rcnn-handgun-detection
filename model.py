import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import numpy as np
import math
import datetime
import os
import re
#######Layers###########
from proposal import ProposalLayer
from training_detection import TrainingDetectionLayer
from inference_detection import InferenceDetectionLayer
from roialign import RoiAlignLayer
from utils import BatchNorm
#######customize########
import resnet101
import RPN
import utils
import losses
import data_generator
########debugs##########
import display_data
from skimage.transform import resize
import cv2

import h5py
from tensorflow.python.keras.engine.saving import hdf5_format



def fpn_classifier(rois, features, image_shape, pool_size, num_classes):
    #ROI pooling + projectation = ROI align
    x = RoiAlignLayer([pool_size,pool_size], image_shape,
                        name="roi_align_classifier")([rois] + features)
    # 2 1024 FCs layers
    #1st layer
    x = layers.TimeDistributed(layers.Conv2D(1024, (pool_size,pool_size), padding="valid"),
                               name="mrcnn_class_conv1")(x)
    x = layers.TimeDistributed(BatchNorm(axis=3), name='mrcnn_class_bn1')(x)
    x = layers.Activation('relu')(x)
    #2nd layer
    x = layers.TimeDistributed(layers.Conv2D(1024, (1,1)), name="mrcnn_class_conv2")(x)
    x = layers.TimeDistributed(BatchNorm(axis=3), name='mrcnn_class_bn2')(x)
    x = layers.Activation('relu')(x)

    shared = layers.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2))(x) #h=w=1 no need that information

    rcnn_class_ids = layers.TimeDistributed(layers.Dense(num_classes),name='mrcnn_class_logits')(shared)
    rcnn_probs = layers.TimeDistributed(layers.Activation('softmax'),name="mrcnn_class")(rcnn_class_ids)

    rcnn_bbox = layers.TimeDistributed(layers.Dense(num_classes * 4, activation='linear'),name='mrcnn_bbox_fc')(shared)
    shape = K.int_shape(rcnn_bbox)
    if shape[1]==None:
        rcnn_bbox = layers.Reshape((-1, num_classes, 4) ,name="mrcnn_bbox")(rcnn_bbox)
    else:
        rcnn_bbox = layers.Reshape((shape[1], num_classes, 4), name="mrcnn_bbox")(rcnn_bbox)
    #rcnn_bbox = layers.Reshape((shape[1],num_classes,4))(rcnn_bbox) #[batch, num_rois, num_class, (dy,dx,log(dh),log(dw))]

    return rcnn_class_ids, rcnn_probs, rcnn_bbox

class AnchorLayers(layers.Layer):
    def __init__(self, name="anchors", **kwargs):
        super(AnchorLayers, self).__init__(name=name, **kwargs)

    def call(self, anchor):
        return anchor

    def get_config(self) :
        config = super(AnchorsLayer, self).get_config()
        return config

class RCNN():
    def __init__(self,mode,config):
        self.mode = mode
        self.config = config #config hyperparameter
        self.model_dir = "D:\My_Code\database\model"
        self.set_log_dir()
        self.rcnn_model = self.build(mode=mode, config=config)

    def set_log_dir(self, path=None):
        self.epoch = 0
        now = datetime.datetime.now()

        if path:
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "rcnn_{}_*epoch*_*val_loss*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")
        self.checkpoint_path = self.checkpoint_path.replace(
            "*val_loss*", "{val_loss:.2f}")

    def build(self,mode,config):

        tf.compat.v1.disable_eager_execution()

        #input
        input_image = keras.Input(shape = config.IMAGE_SHAPE.tolist(), name = "input_image")
        input_image_meta = keras.Input(shape = [None], name = "input_image_meta")

        if mode == "train":
            #RPN
            input_rpn_match = keras.Input(shape = [None,1],name="input_rpn_match", dtype=tf.int32) #match
            input_rpn_bbox = keras.Input(shape = [None,4], name="input_rpn_bbox", dtype=tf.float32) #bounding box

            #ground truth
            input_gt_ids = keras.Input(shape = [None],name="input_gt_class_ids", dtype=tf.int32) #GT class IDs
            input_gt_boxes = keras.Input(shape = [None,4],name="input_gt_boxes", dtype=tf.float32)

            h,w = K.shape(input_image)[1], K.shape(input_image)[2]
            image_scale = K.cast(K.stack([h,w,h,w],axis=0), tf.float32)

            #normalize ground truth boxes
            gt_boxes = layers.Lambda(lambda x : x / image_scale)(input_gt_boxes)

            #mini_mask
            #input_gt_masks = keras.Input(shape = [config.MINI_MASK_SHAPE[0],
            #                                      config.MINI_MASK_SHAPE[1], None],
            #                             name = "input_gt_masks", dtype=bool)


        #resnet layer
        _,C2,C3,C4,C5 = resnet101.build_layers(input_image)
        #FPN
        P5 = layers.Conv2D(256, (1, 1), name='fpn_c5p5')(C5)
        P4 = layers.Add(name="fpn_p4add")([
            layers.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            layers.Conv2D(256, (1, 1), name='fpn_c4p4')(C4)])
        P3 = layers.Add(name="fpn_p3add")([
            layers.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            layers.Conv2D(256, (1, 1), name='fpn_c3p3')(C3)])
        P2 = layers.Add(name="fpn_p2add")([
            layers.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            layers.Conv2D(256, (1, 1), name='fpn_c2p2')(C2)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = layers.Conv2D(256, (3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = layers.Conv2D(256, (3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = layers.Conv2D(256, (3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = layers.Conv2D(256, (3, 3), padding="SAME", name="fpn_p5")(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = layers.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)


        RPN_feature = [P2,P3,P4,P5,P6]
        RCNN_feature = [P2,P3,P4,P5]

        self.anchors = utils.generate_anchors(self.config.ANCHOR_SCALES,
                                         self.config.ANCHOR_RATIOS,
                                         self.config.ANCHOR_STRIDE,
                                         self.config.BACKBONE_SHAPES,
                                         self.config.BACKBONE_STRIDES)

        #RPN Keras model
        input_feature = keras.Input(shape=[None,None,config.PIRAMID_SIZE],name="input_rpn_feature_map")

        """
        rpn_class_cls: anchor class classifier
        rpn_probs: anchor classifier probability
        rpn_bbox_offset: anchor bounding box offset
        """
        outputs = RPN.build_graph(input_feature,len(config.ANCHOR_RATIOS),config.ANCHOR_STRIDE)

        RPN_model = keras.Model([input_feature], outputs, name="rpn_model")

        """
        In FPN, we generate a pyramid of feature maps. We apply the RPN (described in the previous section)
        to generate ROIs. Based on the size of the ROI,
        we select the feature map layer in the most proper scale to extract the feature patches.
        """
        layer_outputs = []
        for x in RPN_feature:
            layer_outputs.append(RPN_model([x]))

        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [layers.Concatenate(axis=1, name = n)(list(o))
                         for o,n in zip(outputs, output_names)]

        #rpn_class_cls, rpn_probs, rpn_bbox = layer_outputs
        rpn_class_ids, rpn_probs, rpn_bbox_offset = outputs


        #Proposal layer
        if mode == "train":
            num_proposal = config.NUM_ROI_TRAINING
        else:
            num_proposal = config.NUM_ROI_INFERENCE

        ROIS_proposals = ProposalLayer(num_proposal=num_proposal, nms_threshold=config.NMS_THRESHOLD,
                                       anchors = self.anchors, config=config)([rpn_probs,rpn_bbox_offset])

        #combine together
        if mode == "train":
            #class ids
            total_class_ids = layers.Lambda(lambda x : utils.parse_image_meta(x)["class_ids"])(input_image_meta)

            #Subsamples proposals and generates target box refinement, class_ids 1.7
            #ratio postive/negative rois = 1/3 (threshold = 0.5)
            #target_ids: class ids of gt boxes closest to positive roi
            #target_bbox = offset from positive rois to it's closest gt_box

            target_rois = ROIS_proposals

            #rois, target_ids, target_bbox, target_mask =\
            # TrainingDetectionLayer(config, name="proposal_targets")([target_rois,
            #                                                          input_gt_ids,gt_boxes, input_gt_masks])

            rois, target_ids, target_bbox =\
             TrainingDetectionLayer(config, name="proposal_targets")([target_rois,
                                                                      input_gt_ids,gt_boxes])

            #classification and regression ROIs after RPN through FPN
            rcnn_class_ids,rcnn_class_probs, rcnn_bbox = fpn_classifier(rois, RCNN_feature, config.IMAGE_SHAPE,
                                                                         config.POOL_SIZE,config.NUM_CLASSES)

            #rcnn_mask = fpn_mask(rois, RCNN_feature, config.IMAGE_SHAPE, config.MASK_POOL_SIZE, config.NUM_CLASSES)

            output_rois = layers.Lambda(lambda x:x * 1, name="output_rois")(rois)

            #rpn losses
            rpn_class_loss = layers.Lambda(lambda x : losses.rpn_class_loss_func(*x), name="rpn_class_loss")(
                             [input_rpn_match, rpn_class_ids])

            rpn_bbox_loss = layers.Lambda(lambda x : losses.rpn_bbox_loss_func(config, *x), name="rpn_bbox_loss")(
                             [input_rpn_bbox, input_rpn_match, rpn_bbox_offset])
            #rcnn losses

            rcnn_class_loss = layers.Lambda(lambda x : losses.rcnn_class_loss_func(*x), name="mrcnn_class_loss")(
                             [target_ids, rcnn_class_ids, total_class_ids])

            rcnn_bbox_loss = layers.Lambda(lambda x : losses.rcnn_bbox_loss_func(*x), name="mrcnn_bbox_loss")(
                             [target_bbox, target_ids, rcnn_bbox])

            #rcnn_mask_loss = layers.Lambda(lambda x : losses.rcnn_mask_loss_func(*x), name="mrcnn_mask_loss")(
            #                 [target_mask, target_ids, rcnn_mask])

            #MODEL
            """
            inputs = [input_image, input_image_meta, input_rpn_match, input_rpn_bbox,
                      input_gt_ids, input_gt_boxes, input_gt_masks]
            outputs = [rpn_class_ids, rpn_probs, rpn_bbox_offset,
                       rcnn_class_ids,rcnn_class_probs, rcnn_bbox, rcnn_mask,
                       ROIS_proposals, output_rois,
                       rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss, rcnn_mask_loss]
            """
            inputs = [input_image, input_image_meta, input_rpn_match, input_rpn_bbox,
                      input_gt_ids, input_gt_boxes]
            outputs = [rpn_class_ids, rpn_probs, rpn_bbox_offset,
                       rcnn_class_ids,rcnn_class_probs, rcnn_bbox,
                       ROIS_proposals, output_rois,
                       rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss]

            model = keras.Model(inputs,outputs,name='mask_rcnn')

        else:
            rcnn_class_ids,rcnn_class_probs, rcnn_bbox =\
                fpn_classifier(ROIS_proposals, RCNN_feature, config.IMAGE_SHAPE,
                               config.POOL_SIZE,config.NUM_CLASSES)

            #[N, (y1, x1, y2, x2, class_id, score)]
            detections = InferenceDetectionLayer(config, name="mrcnn_detection")(
                [ROIS_proposals,rcnn_class_probs,rcnn_bbox,input_image_meta]
            )

            #h,w = config.IMAGE_SHAPE[:2]

            #detection_boxes = layers.Lambda(
            #                  lambda x: x[..., :4] / np.array([h,w,h,w]))(detections)

            inputs = [input_image, input_image_meta]
            outputs = [detections, rcnn_class_probs, rcnn_bbox,
                      ROIS_proposals, rpn_probs, rpn_bbox_offset]

            model = keras.Model(inputs,outputs,name='mask_rcnn')


        #print(model.layers)
        return model

    def load_weights(self, path, by_name, isCoco):

        #if path == "/content/drive/My Drive/database/model/mask_rcnn_balloon.h5":
        if isCoco == 1:
            exclude = ["mrcnn_class_logits", "mrcnn_bbox_fc",
                       "mrcnn_bbox", "mrcnn_mask"]
        else:
            exclude = None

        f = h5py.File(path, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.rcnn_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            hdf5_format.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            hdf5_format.load_weights_from_hdf5_group(f, layers)

        #for l in layers:
            #print(l.get_weights())

        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(path)

    def find_last(self):

        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir))
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def set_trainable(self,layers_regex, model=None, indent=0, verbose=1):
        """
        trainable_weights is the list of those that are meant to be updated
         (via gradient descent) to minimize the loss during training.
         non_trainable_weights is the list of those that aren't meant to be trained.
         Typically they are updated by the model during the forward pass.
        """

        model = model or self.rcnn_model
        layers = model.layers

        for layer in layers:
            #print(layer.name,layer.__class__.__name__)
            if layer.__class__.__name__ == "Model":
                #print("in model: ",layer.name)
                self.set_trainable(layers_regex, model = layer, indent = indent+4)
                continue

            if not layer.weights:
                continue

            trainable = bool(re.fullmatch(layers_regex, layer.name))

            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable

            #print trainable layers
            if trainable and verbose > 0:
                print("{}{:20}   ({})".format(" " * indent, layer.name,layer.__class__.__name__))

    def compile(self,learning_rate, momentum):
        optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum,clipnorm=self.config.GRADIENT_CLIP_NORM)

        #self.rcnn_model._losses = []
        #self.rcnn_model._per_input_losses = []

        """
        loss_func_name = ["rpn_class_loss","rpn_bbox_loss",
                          "mrcnn_class_loss","mrcnn_bbox_loss","mrcnn_mask_loss"]
        """

        loss_func_name = ["rpn_class_loss","rpn_bbox_loss",
                          "mrcnn_class_loss","mrcnn_bbox_loss"]

        for name in loss_func_name:
            layer = self.rcnn_model.get_layer(name)

            if layer.output in self.rcnn_model.losses:
                continue

            loss = (
                tf.math.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.rcnn_model.add_loss(loss)

        #l2 Regularization
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.rcnn_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name
        ]

        self.rcnn_model.add_loss(tf.add_n(reg_losses))

        self.rcnn_model.compile(optimizer=optimizer, loss=[None] * len(self.rcnn_model.outputs))

        for name in loss_func_name:
            if name in self.rcnn_model.metrics_names:
                continue

            layer = self.rcnn_model.get_layer(name)
            self.rcnn_model.metrics_names.append(name)

            loss = (
                tf.math.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.))

            self.rcnn_model.add_metric(loss,name=name, aggregation='mean')


    def train(self, dataset_train, dataset_val, learning_rate, epochs):
        print("START TRAINING!")

        #for l in self.rcnn_model.layers:
        #    print(l.name)

        train_generator = data_generator.gen(dataset_train, self.config, shuffle=True, batch_size=self.config.BATCH_SIZE)
        val_generator = data_generator.gen(dataset_val, self.config, shuffle=True, batch_size=self.config.BATCH_SIZE)

        #display_data.view(dataset_train, self.config, shuffle=True, batch_size=self.config.BATCH_SIZE)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        callbacks = [
            keras.callbacks.TensorBoard(log_dir = self.log_dir,histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path, verbose=0, save_weights_only=True)
        ]

        print("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        print("Checkpoint Path: {}".format(self.checkpoint_path))

        #set layers trainable
        #self.set_trainable()

        #since already load pretrained model, so we don't need train backbone
        layers = r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)"
        #layers=r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)"
        #layers = ".*"
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        workers = 0

        self.rcnn_model.fit(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True
        )

        self.epoch = max(self.epoch, epochs)

    def mold_inputs(self, images):
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image to fit the model expected size
            # TODO: move resizing to mold_image()

            molded_image = resize(image, (self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM))

            shape = molded_image.shape
            molded_image = molded_image * np.full((shape),255.0)

            molded_image = utils.mold_image(molded_image, self.config)

            window = (0,0,self.config.IMAGE_MAX_DIM,self.config.IMAGE_MAX_DIM)
            # Build image_meta
            image_meta = utils.compose_image_meta(
                0, molded_image.shape, window,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        #print(molded_images.shape)
        #print(image_metas.shape)
        #print(windows.shape)
        return molded_images, image_metas, windows


    def unmold_detections(self, detections, image_shape, window):
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]

        """
        # Compute scale and shift to translate coordinates to image domain.
        h_scale = image_shape[0] / (window[2] - window[0])
        w_scale = image_shape[1] / (window[3] - window[1])
        scale = min(h_scale, w_scale)
        shift = window[:2]  # y, x
        scales = np.array([scale, scale, scale, scale])
        shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

        # Translate bounding boxes to image domain
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

        # Filter out detections with zero area. Often only happens in early
        # stages of training when the network weights are still a bit random.
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        """
        return boxes, class_ids, scores

    def detect(self, image, verbose=0):
        if verbose:
            print("processing image....")

        molded_image, image_meta, window = self.mold_inputs(image)

        print("molded_image ",molded_image.shape)
        print("image_meta",image_meta)

        detection, rcnn_class_probs, rcnn_bbox,\
            ROIS_proposals, rpn_probs, rpn_bbox_offset = \
                self.rcnn_model.predict([molded_image,image_meta], verbose=0)

        """
        f = open("todo.txt","w")
        for x in ROIS_proposals[0]:
            x *= np.array([896,896,896,896])
            f.write(str(x))

        for x in detection[0]:
            #x *= np.array([832,832,832,832])
            f.write(str(x))
        f.close()
        """
        """
        rois = []
        for x in ROIS_proposals[0]:
            x *= np.array([896,896,896,896])
            rois.append(x)

        display_data.inspect(image[0],rois)
        """

        #for x in rcnn_class_probs[0]:
        #    print(x)

        #print(rcnn_bbox)
        results = []

        for i,img in enumerate(image):
            final_rois, final_class_ids, final_scores =\
                self.unmold_detections(detection[i], img.shape, window[i])

            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
            })

        print(results[0]['class_ids'])
        print(results[0]['scores'])

        ROI = []

        for i,img in enumerate(image):
            """
            img = resize(img,(self.config.IMAGE_MAX_DIM,self.config.IMAGE_MAX_DIM))
            shape = img.shape
            img = img * np.full((shape),255.0)
            """

            shape = img.shape

            for roi in results[i]['rois']:
                y1,x1,y2,x2 = roi

                x1 = x1 * 1.0 * shape[1] / self.config.IMAGE_MAX_DIM
                x2 = x2 * 1.0 * shape[1] / self.config.IMAGE_MAX_DIM
                y1 = y1 * 1.0 * shape[0] / self.config.IMAGE_MAX_DIM
                y2 = y2 * 1.0 * shape[0] / self.config.IMAGE_MAX_DIM


                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)

                ROI.append([y1,x1,y2,x2])

        return ROI, results[0]['scores']

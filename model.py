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

def fpn_classifier(rois, features, image_meta, pool_size, num_classes, train_bn=True, fc_layers_size = 1024):
    #ROI pooling + projectation = ROI align
    x = RoiAlignLayer([pool_size,pool_size])([rois,image_meta] + features)
    # 2 1024 FCs layers
    #1st layer
    x = layers.TimeDistributed(layers.Conv2D(fc_layers_size, (pool_size,pool_size), padding="valid"), name="mrcnn_class_conv1")(x)
    x = layers.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = layers.Activation('relu')(x)
    #2nd layer
    x = layers.TimeDistributed(layers.Conv2D(fc_layers_size, (1,1), padding="valid"), name="mrcnn_class_conv2")(x)
    x = layers.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = layers.Activation('relu')(x)

    shared = layers.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2))(x) #h=w=1 no need that information

    rcnn_class_ids = layers.TimeDistributed(layers.Dense(num_classes),name='mrcnn_class_logits')(shared)
    rcnn_probs = layers.TimeDistributed(layers.Activation('softmax'),name="mrcnn_class")(rcnn_class_ids)

    rcnn_bbox = layers.TimeDistributed(layers.Dense(num_classes * 4, activation='linear'),name='mrcnn_bbox_fc')(shared)
    shape = K.int_shape(rcnn_bbox)
    if shape[1]==None:
        rcnn_bbox = layers.Reshape((-1, num_classes, 4))(rcnn_bbox)
    else:
        rcnn_bbox = layers.Reshape((shape[1], num_classes, 4))(rcnn_bbox)
    #rcnn_bbox = layers.Reshape((shape[1],num_classes,4))(rcnn_bbox) #[batch, num_rois, num_class, (dy,dx,log(dh),log(dw))]

    return rcnn_class_ids, rcnn_probs, rcnn_bbox


class RCNN():
    def __init__(self,mode,config):
        self.mode = mode
        self.config = config #config hyperparameter
        self.anchor_cache = {} #dict
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

    def get_anchors(self,image_shape):

        backbone_shape = np.array([[int(math.ceil(image_shape[0]/stride)),int(math.ceil(image_shape[1]/stride))] for stride in self.config.BACKBONE_STRIDES])

        if not tuple(image_shape) in self.anchor_cache:
            anchors = utils.generate_anchors(self.config.ANCHOR_SCALES,
                                             self.config.ANCHOR_RATIOS,
                                             self.config.ANCHOR_STRIDE,
                                             backbone_shape,
                                             self.config.BACKBONE_STRIDES)

            self.anchor_cache[tuple(image_shape)] = utils.norm_boxes(anchors,image_shape[:2])

        return self.anchor_cache[tuple(image_shape)]

    def build(self,mode,config):
        h,w = config.IMAGE_SHAPE[:2]
        model = None

        #input
        input_image = keras.Input(shape = [None,None,config.IMAGE_SHAPE[2]], name = "input_image")
        input_image_meta = keras.Input(shape = [config.IMAGE_META_SIZE], name = "input_image_meta")

        #resnet layer
        C1,C2,C3,C4,C5 = resnet101.build_layers(input = input_image, train_bn=config.TRAIN_BN)
        #FPN
        P2,P3,P4,P5,P6 = resnet101.build_FPN(C1=C1,C2=C2,C3=C3,C4=C4,C5=C5,config=config)

        RPN_feature = [P2,P3,P4,P5,P6]
        RCNN_feature = [P2,P3,P4,P5]


        if mode == "train":
            #RPN
            input_rpn_match = keras.Input(shape = [None,1],name="input_rpn_match", dtype=tf.int32) #match
            input_rpn_bbox = keras.Input(shape = [None,4], name="input_rpn_bbox", dtype=tf.float32) #bounding box

            #ground truth
            input_gt_ids = keras.Input(shape = [None],name="input_gt_class_ids", dtype=tf.int32) #GT class IDs
            input_gt_boxes = keras.Input(shape = [None,4],name="input_gt_boxes", dtype=tf.float32)

            #normalize ground truth boxes
            gt_boxes = layers.Lambda(lambda x : utils.norm_boxes(x,K.shape(input_image)[1:3]))(input_gt_boxes)

            #anchors for RPN
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            #anchors = layers.Lambda(lambda x : tf.Variable(anchors))(input_image)

        elif mode == "inference":
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")
            anchors = input_anchors

        #RPN Keras model
        input_feature = keras.Input(shape=[None,None,config.PIRAMID_SIZE])

        """
        rpn_class_cls: anchor class classifier
        rpn_probs: anchor classifier probability
        rpn_bbox_offset: anchor bounding box offset
        """
        outputs = RPN.build_graph(input_feature,len(config.ANCHOR_RATIOS),config.ANCHOR_STRIDE)

        RPN_model = keras.Model(inputs=[input_feature], outputs = outputs, name="rpn_model")

        """
        In FPN, we generate a pyramid of feature maps. We apply the RPN (described in the previous section)
        to generate ROIs. Based on the size of the ROI,
        we select the feature map layer in the most proper scale to extract the feature patches.
        """
        layer_outputs = []
        for x in RPN_feature:
            layer_outputs.append(RPN_model([x]))

        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        layer_outputs = list(zip(*layer_outputs))
        layer_outputs = [layers.Concatenate(axis=1, name = n)(list(o)) for o,n in zip(layer_outputs, output_names)]

        #rpn_class_cls, rpn_probs, rpn_bbox = layer_outputs
        rpn_class_ids, rpn_probs, rpn_bbox_offset = layer_outputs


        #Proposal layer
        num_proposal=0
        if mode == "train":
            num_proposal = config.NUM_ROI_TRAINING
        elif mode == "inference":
            num_proposal = config.NUM_ROI_INFERENCE

        ROIS_proposals = ProposalLayer(num_proposal=num_proposal, nms_threshold=config.NMS_THRESHOLD, config=config)([rpn_probs,rpn_bbox_offset,anchors])

        #combine together
        if mode == "train":
            #class ids
            total_class_ids = layers.Lambda(lambda x : utils.parse_image_meta(x)["class_ids"])(input_image_meta)

            #Subsamples proposals and generates target box refinement, class_ids 1.7
            #ratio postive/negative rois = 1/3 (threshold = 0.5)
            #target_ids: class ids of gt boxes closest to positive roi
            #target_bbox = offset from positive rois to it's closest gt_box
            rois, target_ids, target_bbox = TrainingDetectionLayer(config)([ROIS_proposals,input_gt_ids,gt_boxes])

            #classification and regression ROIs after RPN through FPN
            rcnn_class_ids,rcnn_class_probs, rcnn_bbox = fpn_classifier(rois, RCNN_feature, input_image_meta,
                                                                         config.POOL_SIZE,config.NUM_CLASSES,
                                                                         train_bn=config.TRAIN_BN,
                                                                         fc_layers_size=config.FPN_CLS_FC_LAYERS)
            output_rois = layers.Lambda(lambda x:x * 1)(rois)

            #rpn losses
            rpn_class_loss = layers.Lambda(lambda x : losses.rpn_class_loss_func(*x), name="rpn_class_loss")([input_rpn_match, rpn_class_ids])
            rpn_bbox_loss = layers.Lambda(lambda x : losses.rpn_bbox_loss_func(config, *x), name="rpn_bbox_loss")([input_rpn_bbox, input_rpn_match, rpn_bbox_offset])
            #rcnn losses
            rcnn_class_loss = layers.Lambda(lambda x : losses.rcnn_class_loss_func(*x), name="mrcnn_class_loss")([target_ids, rcnn_class_ids, total_class_ids])
            rcnn_bbox_loss = layers.Lambda(lambda x : losses.rcnn_bbox_loss_func(*x), name="mrcnn_bbox_loss")([target_bbox, target_ids, rcnn_bbox])

            #MODEL
            inputs = [input_image, input_image_meta, input_rpn_match, input_rpn_bbox, input_gt_ids, input_gt_boxes]
            outputs = [rpn_class_ids, rpn_probs, rpn_bbox_offset,
                       rcnn_class_ids,rcnn_class_probs, rcnn_bbox,
                       ROIS_proposals, output_rois,
                       rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss]

            model = keras.Model(inputs,outputs,name='mask_rcnn')

        elif mode =="inference":
            rcnn_class_ids,rcnn_class_probs, rcnn_bbox = fpn_classifier(ROIS_proposals, RCNN_feature, input_image_meta,
                                                                         config.POOL_SIZE,self.NUM_CLASSES,
                                                                         train_bn=config.TRAIN_BN,
                                                                         fc_layers_size=config.FPN_CLS_FC_LAYERS)
            #detection
            detections = InferenceDetectionLayer(config)([ROIS_proposals,rcnn_class_probs,rcnn_bbox, input_image_meta])

            inputs = [input_image, input_image_meta, input_anchors]
            output = [detections,
                      rcnn_class_probs, rcnn_bbox,
                      ROIS_proposals,rcnn_class_probs, rcnn_bbox]

            model = keras.Model(inputs,outputs,name='mask_rcnn')

        #print(model.layers)
        return model

    def load_weights(self, path, by_name):
        """
        import h5py

        f = h5py.File(path, "r")
        for k in f.keys():
            print(k)
            print(f[k])
        """

        model = self.rcnn_model
        model.load_weights(path, by_name=by_name)
        print("done load pretrained coco model weights")

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

        self.rcnn_model._losses = []
        self.rcnn_model._per_input_losses = []

        loss_func_name = ["rpn_class_loss","rpn_bbox_loss","mrcnn_class_loss","mrcnn_bbox_loss"]

        for name in loss_func_name:
            layer = self.rcnn_model.get_layer(name)

            if layer.output in self.rcnn_model.losses:
                continue

            loss = (tf.reduce_mean(layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(name,1.))
            self.keras_model.add_loss(loss)

        #l2 Regularization
        reg_losses = [
            keras.regularizers.L2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.rcnn_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name
        ]

        self.keras_model.add_loss(tf.add_n(reg_losses))

        self.rcnn_model.compile(optimizer=optimizer, loss=[None] * len(self.rcnn_model.outputs))

        for name in loss_func_name:
            if name in self.rcnn_model.metrics_names:
                continue

            layer = self.rcnn_model.get_layer(name)
            self.rcnn_model.metrics_names.append(name)

            loss = (tf.reduce_mean(layer.output, keepdims = True) * self.config.LOSS_WEIGHTS.get(name,1.))
            self.rcnn_model.metrics_tensors.append(loss)


    def train(self, dataset_train, dataset_val, learning_rate, epochs):
        print("START TRAINING!")

        #for l in self.rcnn_model.layers:
        #    print(l.name)

        train_generator = data_generator.gen(dataset_train, self.config, shuffle=True, batch_size=self.config.BATCH_SIZE)
        val_generator = data_generator.gen(dataset_val, self.config, shuffle=True, batch_size=self.config.BATCH_SIZE)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        callbacks = [
            keras.callbacks.TensorBoard(log_dir = self.log_dir,histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path, verbose=0, save_weights_only=True)
        ]

        print("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        print("Checkpoint Path: {}".format(self.checkpoint_path))

        """
        print(self.log_dir)
        print(self.checkpoint_path)

        f = open("D:\My_Code\gun_detection\\todo.txt","w")

        for l in self.rcnn_model.layers:
            f.write("\n fuck "+l.name+"\n")
            f.write(str(l.get_weights()))

        f.close()
        """
        #set layers trainable
        #self.set_trainable()

        #since already load pretrained model, so we don't need train backbone
        layers = r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)"
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

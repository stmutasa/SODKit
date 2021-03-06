"""
SOD_ObjectDetection contains the class utilized in implementing the object detector networks based on
Faster-RCNN
Mask-RCNN
"""


from SODNetwork import tf
from SODNetwork import np
from SODNetwork import SODMatrix
from SOD_ResNet import ResNet
from SOD_DenseNet import DenseNet
import tensorflow.contrib.slim as slim

import cv2


class MRCNN(SODMatrix):

    """
    The multiple inheritence class to perform all functions of making mask RCNNs!
    """

    # Shared class variables here

    def __init__(self, phase_train, GPU_count=1, Images_per_gpu=2, FC_nodes=256, K1=32, num_classes=1, FPN_layers=64,
                 RPN_anchor_scales=(1, 0.5), RPN_anchor_ratios=[0.5, 1, 2], image_size=512, RPN_base_anchor_size=[6, 13, 20, 35],
                 RPN_nms_upper_threshold=0.7, RPN_nms_lower_threshold=0.3, RPN_anchors_per_image=256, RPN_class_loss_weight=1.0,
                 batch_size=8, max_proposals=32, RPN_batch_size=256, RPN_batch_positives_ratio=0.5, RPN_box_loss_weight=1.0, PRE_NMS_top_k=8192,
                 POST_NMS_ROIS_training=512, POST_NMS_ROIS_testing=256, Use_mini_mask=False, Mini_mask_shape=[56, 56], FRCNN_positives_threshold=0.5,
                 FRCNN_batch_size=256, FRCNN_batch_positives_ratio=0.5, FRCNN_box_loss_weight=1.0, FRCNN_class_loss_weight=1.0):

        """
        Instance specific variables

        :param phase_train: Training or testing mode
        :param FC_nodes: Number of neurons in the first fully connected layer
        :param K1: Number of filters in the first layer
        :param num_classes: Number of classification classes. Does not include the background category
        :param FPN_layers: The number of layers in the FPN outputs

        :param image_size: The size of the input images
        :param batch_size: The number of input images processed at a time
        :param GPU_count: Number of GPUs to use
        :param Images_per_gpu: Number of images to train with on each GPU. Use highest number gpu can handle

        :param RPN_anchor_scales: Length of square anchor side in pixels
        :param RPN_anchor_ratios: Ratios of anchors at each cell (width/height) 1 represents a square anchor, and 0.5 is a wide anchor
        :param RPN_base_anchor_size: The list of anchor sizes for [32, 64, 128, 256] feature map sizes
        :param RPN_nms_upper_threshold: Non-max suppression threshold to filter RPN proposals. Increase for more proposals
        :param RPN_nms_lower_threshold: Non-min suppression threshold to filter RPN proposals. Decrease for more proposals
        :param RPN_anchors_per_image: how many anchors per image to use for RPN training
        :param RPN_batch_size: The batch size of the region proposal network otuput
        :param RPN_batch_positives_ratio: The ratio of positives in the RPN output batch
        :param RPN_class_loss_weight: Relative weighting of the RPN foreground loss
        :param RPN_box_loss_weight: relative weighting of the RPN bounding box loss

        :param FRCNN_positives_threshold: Threshold iou to declare a positive detection on the faster RCNN head

        :param POST_NMS_ROIS_training: ROIs kept after non-maximum supression (training)
        :param POST_NMS_ROIS_testing: ROIs kept after non-maximum supression (inference)
        :param max_proposals: The number of proposals output by the RPN during testing or training
        :param PRE_NMS_top_k: Limit the # of RPN outputs that get sent to NMS to this number. Must be far larger than max_proposals

        :param Use_mini_mask: If enabled, resizes instance masks to a smaller size to reduce memory usage
        :param Mini_mask_shape: (height, width) of the mini-mask
        """

        # Baseline parameters for FPN
        self.phase_train = phase_train
        self.GPU_count = GPU_count
        self.Images_per_gpu = Images_per_gpu
        self.FC_nodes = FC_nodes
        self.K1 = K1
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.image_size = image_size
        self.FPN_layers = FPN_layers

        # Region proposal network parameters
        self.RPN_anchor_scales = RPN_anchor_scales
        self.RPN_anchor_ratios = RPN_anchor_ratios
        self.RPN_nms_upper_threshold = RPN_nms_upper_threshold
        self.RPN_nms_lower_threshold = RPN_nms_lower_threshold
        self.RPN_anchors_per_image = RPN_anchors_per_image
        self.RPN_batch_positives_ratio = RPN_batch_positives_ratio
        self.RPN_batch_size = RPN_batch_size

        # Faster RCNN head parameters
        self.FRCNN_positives_threshold = FRCNN_positives_threshold
        self.FRCNN_batch_positives_ratio = FRCNN_batch_positives_ratio
        self.FRCNN_batch_size = FRCNN_batch_size

        # RPN Proposal variables
        self.POST_NMS_ROIS_training = POST_NMS_ROIS_training
        self.POST_NMS_ROIS_testing = POST_NMS_ROIS_testing
        self.max_proposals = max_proposals
        self.PRE_NMS_top_k = PRE_NMS_top_k

        self.Use_mini_mask = Use_mini_mask
        self.Mini_mask_shape = Mini_mask_shape
        self.RPN_base_anchor_size = RPN_base_anchor_size

        # Loss function weights
        self.RPN_class_loss_weight = RPN_class_loss_weight
        self.RPN_box_loss_weight = RPN_box_loss_weight
        self.FRCNN_box_loss_weight = FRCNN_box_loss_weight
        self.FRCNN_class_loss_weight = FRCNN_class_loss_weight

        # Keeping track of the layers and losses
        # self.RPN_ROI = None
        # self.RPN_Loss_Object = None
        # self.RPN_Loss_Box = None
        # self.RPN_class_logits = None
        # self.RPN_bbox_score = None
        # self.Anchors = None
        # self.gt_boxes = None

    """
    Feature Pyramid Network Functions
    """

    def Generate_FPN(self, input_images, net_type, input_dims, FPN=1):

        """
        Builds the base feature pyramid network
        :param input_images: the input images
        :param net_type: can be a ResNet, DenseNet or Inception net
        :param input_dims: dimensions of the input images
        :param FPN: Whether to use a feature pyramid network (you should). 0 = no, 1 = 2D, 2 = 2.5 D
        :return:
                The final feature map outputs from the feature pyramid network
        """

        self.Image_batch = input_images

        # Calculate how many blocks are needed to get to a final dimension of 32x32
        nb_blocks = int((input_dims / 32) ** (0.5)) + 1

        # Calculate block layers downsampling scheme
        block_sizes = [None] * nb_blocks
        for z in range(nb_blocks): block_sizes[z] = 2 + 2 * z
        block_sizes[0] = 1

        if net_type == 'RESIDUAL':

            # Define a ResNet
            self.resnet = ResNet(nb_blocks=nb_blocks, filters=self.K1, images=input_images, sess=None, phase_train=self.phase_train, summary=True)

            # Make sure all the 64x64 and below feature maps are inception style
            inception_layers = [None] * nb_blocks
            inception_layers[-1], inception_layers[-2] = 1, 1

            # Define the downsample network and retreive the output of each block #TODO: Fix class inheritence and FPN layers
            if FPN == 1: _, self.conv = self.resnet.define_network(block_sizes, inception_layers, FPN=True, FPN_layers=self.FPN_layers)
            elif FPN == 2: _, self.conv = self.resnet.define_network_25D(block_sizes, inception_layers, FPN=True, FPN_layers=self.FPN_layers)
            else: _, self.conv = self.resnet.define_network(block_sizes, inception_layers, FPN=False)

        if net_type == 'DENSE':
            pass


    """
    Region Proposal Network Functions
    """


    def RPN_Process(self):

        """
        Makes anchors and runs the RPN forward pass
        TODO: Prune anchors outside image window at train. Clip anchors to image window at test time.
        :return:
        """

        # Run the forward pass and generate anchor boxes: These return as [batch, n, x] with n = # of anchors
        class_logits, box_logits = self._RPN_conv()
        anchors = self._make_anchors_FPN()

        # Also keep a list of which image each anchor came from. Use array broadcasting
        batch_img = tf.cast(tf.lin_space(0.0, tf.cast(self.batch_size-1, tf.float32), self.batch_size), tf.int16)
        anchor_img_source = tf.transpose(tf.multiply(batch_img, tf.transpose(tf.ones_like(anchors[:, :, -1:], tf.int16))))

        # Reshape the batched anchors to a single list:
        class_logits, box_logits = tf.reshape(class_logits, [-1, 2]), tf.reshape(box_logits, [-1, 4])
        anchors = tf.reshape(anchors, [-1, 4])
        self.anchor_img_source = tf.reshape(anchor_img_source, [-1, 1])

        # Under new name scope (not get_variable), clean up the anchors generated during training
        with tf.name_scope('RPNN_Forward'):
            if self.training_phase is True:

                # Remove outside anchor boxes by returning only the indices of the valid anchors
                valid_indices = self._filter_outside_anchors(anchors, self.image_size)
                valid_anchors = tf.gather(anchors, valid_indices)
                valid_cls_logits, valid_box_logits = tf.gather(class_logits, valid_indices), tf.gather(box_logits, valid_indices)

                # Return the valid anchors, else during training just return the anchors
                self.anchors, self.RPN_class_logits, self.RPN_box_logits = valid_anchors, valid_cls_logits, valid_box_logits

            else:
                self.anchors, self.RPN_class_logits, self.RPN_box_logits = anchors, class_logits, box_logits


    def RPN_Post_Process(self, summary=None):

        """
        This function first creates a minibatch by attempting to combine equal positive and negative box proposals
        Then it calculates the losses for the minibatch box dimensions and classifications of background/foreground
        :param summary: Whether to print a summary image to tensorboard
        :return:
        """

        with tf.variable_scope('RPN_Losses'):

            # Run the post processing functions to retreive the gneerated minibatches from the RPN
            minibatch_indices, minibatch_anchor_matched_gtboxes, object_mask, minibatch_labels_one_hot = self._RPN_make_minibatch(self.anchors)
            negative_object_mask = tf.cast(tf.logical_not(tf.cast(object_mask, tf.bool)), tf.float32)

            # Gather the anchors and logits that made it into the generated minibatch
            self.minibatch_image_source = tf.gather(self.anchor_img_source, minibatch_indices)
            minibatch_anchors = tf.gather(self.anchors, minibatch_indices)
            minibatch_class_logits = tf.gather(self.RPN_class_logits, minibatch_indices)
            minibatch_box_logits = tf.gather(self.RPN_box_logits, minibatch_indices)

            # Calculate deltas required to transform anchors to gtboxes, aka the loss
            minibatch_encoded_gtboxes = self._find_deltas(minibatch_anchor_matched_gtboxes, minibatch_anchors)
            minibatch_decoded_boxes = self._apply_deltas(minibatch_encoded_gtboxes, minibatch_anchors) # For the image summary

            # TODO: if we want to generate a summary image
            if summary:
                # # Draw tensorboard summary boxes
                # positive_anchors_in_img = draw_box_with_color()
                # negative_anchors_in_img = draw_box_with_color()
                # tf.summary.image('/positive_anchors', positive_anchors_in_img)
                # tf.summary.image('/negative_anchors', negative_anchors_in_img)
                pass

            # Now for the losses
            with tf.variable_scope('rpn_location_loss'):

                # First calculate the smooth L1 location loss and save to the loss collection
                location_loss = self._smooth_l1_loss(minibatch_box_logits, minibatch_encoded_gtboxes, object_mask)
                location_loss *= self.RPN_box_loss_weight
                tf.add_to_collection('losses', location_loss)

            # Now calculate softmax loss and save
            with tf.variable_scope('rpn_classifcation_loss'):

                classification_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=minibatch_labels_one_hot, logits=minibatch_class_logits)
                classification_loss = tf.reduce_mean(classification_loss)
                classification_loss *= self.RPN_class_loss_weight
                tf.add_to_collection('losses', classification_loss)

            self.RPN_location_loss = location_loss
            self.RPN_classification_loss = classification_loss


    def RPN_Get_Proposals(self):

        """
        This function runs for training and testing. It sends the regions that the RPN detects as object through to the Fast RCNN
        :return:
        """

        with tf.variable_scope('RPN_Proposals'):

            # Apply the box logit adjustments to the anchor boxes TODO: This way is inefficient
            rpn_adjusted_boxes = self._apply_deltas(self.RPN_box_logits, self.anchors)

            # Get softmax scores of the rpn boxes. Retreive just the score for the object column
            rpn_softmax_scores = slim.softmax(self.RPN_class_logits)
            rpn_object_score = rpn_softmax_scores[:, 1]
            rpn_img_fpn_source = tf.concat([self.anchor_img_source, self.anchor_fpn_sources], axis=1)

            # During testing, clip abberant box adjustments to the image boundaries.
            # TODO: Add in the image sources somehow here... Probably just copy again
            # TODO: may actually have to do this for early training phase
            if self.training_phase is False:
                rpn_adjusted_boxes = self._clip_boxes_to_img_boundaries(rpn_adjusted_boxes, self.image_size)

            # Limit the number of proposals that get sent to NMS
            rpn_object_score, top_k_indices = tf.nn.top_k(rpn_object_score, k=self.PRE_NMS_top_k)
            rpn_adjusted_boxes = tf.gather(rpn_adjusted_boxes, top_k_indices)
            rpn_img_fpn_source = tf.gather(rpn_img_fpn_source, top_k_indices)

            # Perform the NMS to return indices of single predictions. Test and train will output different values
            if self.training_phase is False:
                valid_indices = self._non_max_suppression(rpn_adjusted_boxes, rpn_object_score,
                                max_output_size=self.POST_NMS_ROIS_testing, iou_threshold=self.RPN_nms_upper_threshold)

            else:
                valid_indices = self._non_max_suppression(rpn_adjusted_boxes, rpn_object_score,
                                max_output_size=self.POST_NMS_ROIS_testing, iou_threshold=self.RPN_nms_upper_threshold)

            # Retreive the actual the adjusted anchors/predictions/images that are A: in PRE_NMS_Top_k object predictions and B: Passed NMS with threshold iou
            valid_boxes = tf.gather(rpn_adjusted_boxes, valid_indices)
            valid_scores = tf.gather(rpn_object_score, valid_indices)
            valid_sources = tf.gather(rpn_img_fpn_source, valid_indices)

            # Draw boxes
            proposal_images = self.draw_box_in_img_batch(self.Image_batch, valid_boxes, valid_sources[:, 0])
            for z in range(len(proposal_images)): tf.summary.image('/RPN_proposals', proposal_images[z], max_outputs=self.batch_size)

            # If not enough boxes are left, pad with zeros
            rpn_proposal_boxes, rpn_proposal_scores, rpn_proposal_sources = tf.cond(tf.less(tf.shape(valid_boxes)[0], self.max_proposals),
                    lambda: self._pad_boxes_zeros(valid_boxes, valid_scores, valid_sources, self.max_proposals),
                    lambda: (valid_boxes, valid_scores, valid_sources))

            return rpn_proposal_boxes, rpn_proposal_scores, rpn_proposal_sources

    """
    Fast RCNN Network Functions
    """


    def Fast_RCNN(self, rpn_proposal_boxes, rpn_proposal_scores, rpn_proposal_sources):

        """
        Receives non normalized boxes and Performs ROI align using built in tensorflow function
        :param rpn_proposal_boxes: [n, 4] The proposals generated by the RPN
        :param rpn_proposal_scores: [n, ] The objectness scores from the RPN
        :param rpn_proposal_sources: [n, batch#, FPN_dim] the image and feature pyramid source of each proposal
        :return:
        """

        # Carry over RPN Variables
        self.rpn_proposal_boxes, self.rpn_proposal_scores, self.rpn_proposal_sources = \
            rpn_proposal_boxes, rpn_proposal_scores, rpn_proposal_sources

        # Perform ROI align
        self._ROI_Align()

        # Perform the convolution
        self._FRCNN_conv()

        # Process proposals, make minibatches, and retreive loss
        self._FRCNN_loss()


    def Fast_RCNN_Predict(self):

        """
        Run an RCNN forward pass and NMS for testing
        :return:
        """

        with tf.variable_scope('FRCNN_Predict'):

            # Normalize the scores
            FRCNN_softmax_scores = tf.nn.softmax(self.FRCNN_class_logits)

            # Boxes
            FRCNN_boxes = tf.reshape(self.FRCNN_box_logits, [-1, 4])

            # Reference boxes. TODO: batchify this
            FRCNN_reference_boxes = tf.tile(self.gt_boxes, [1, self.num_classes])
            FRCNN_reference_boxes = tf.reshape(FRCNN_reference_boxes, [-1, 4])

            # Encode and clip the boxes
            FRCNN_clipped_boxes = self._find_deltas(FRCNN_boxes, FRCNN_reference_boxes)
            FRCNN_clipped_boxes = self._clip_boxes_to_img_boundaries(FRCNN_clipped_boxes, self.image_size)

            # Multiclass NMS
            FRCNN_nms_boxes = tf.reshape(FRCNN_clipped_boxes, [-1, self.num_classes*4])
            FRCNN_nms_boxes, FRCNN_score, num_objects, detection_category = self._FRCNN_proposals_only()

            return FRCNN_nms_boxes, FRCNN_score, num_objects, detection_category


    """
    RPN hidden Inside functions
    """

    def _RPN_conv(self):

        """
        TODO: Defines the region proposal network
        :return:
                ROI: Regions of interest
                RPN_Loss_Object: The loss regaarding whether an object was found
                RPN_Loss_Box: The loss regarding the bounding box
        """

        # Create arrays to hold the output logits for each level
        class_logits, box_logits = [], []

        # Calculate number of outputs for each head
        try: num_class_scores = 2 * len(self.RPN_anchor_scales) * len(self.RPN_anchor_ratios)
        except: num_class_scores = 2 * len(self.RPN_anchor_ratios)
        num_bbox_scores = 2 * num_class_scores

        # Loop through the FPN levels
        for lvl in range(len(self.conv)):

            # To share the head, we need reuse flag and a scope list
            reuse_flag = None if lvl == 0 else True
            scope_list = ['RPN_3x3', 'RPN_Classifier', 'RPN_Regressor']

            # Now run a 3x3 conv, then separate 1x1 convs for each feature map #TODO: Check if VALID, check BN/Relu
            rpn_3x3 = self._convolution_RPN(scope_list[0], self.conv[lvl], 3, self.FPN_layers, phase_train=self.phase_train, reuse=reuse_flag)
            rpn_class = self._convolution_RPN(scope_list[1], rpn_3x3, 1, num_class_scores, BN=False, relu=False, bias=False, phase_train=self.phase_train, reuse=reuse_flag)
            rpn_box = self._convolution_RPN(scope_list[2], rpn_3x3, 1, num_bbox_scores, BN=False, relu=False, bias=False, phase_train=self.phase_train, reuse=reuse_flag)

            # Reshape the scores and append to the list. Test: Reshaped with batch size
            rpn_class, rpn_box = tf.reshape(rpn_class, [self.batch_size, -1, 2]), tf.reshape(rpn_box, [self.batch_size, -1, 4])
            class_logits.append(rpn_class)
            box_logits.append(rpn_box)

        # Test: Concat along axis 2
        cll, bll = tf.concat(class_logits, axis=1), tf.concat(box_logits, axis=1)

        self.t1, self.t2, self.t3 = class_logits[0], cll, bll

        # Return the concatenated list
        return cll, bll

    def _make_anchors_FPN(self):

        """
        Makes anchors from all levels of the feature pyramid network
        :return:
        """

        # Var scope creates a unique scope for all variables made including with tf.get_variable. name_scope allows reusing of variables
        with tf.variable_scope('make_anchors'):

            # List of anchors
            anchor_list, FPN_sources = [], []

            # Name scope doesn't rescope tf.get_variable calls
            with tf.name_scope('make_anchors_all_levels'):

                # Loop through each FPN feature map scale
                for FPN_scale, base_anchor_size in zip(self.conv, self.RPN_base_anchor_size):

                    # Stride = usually 1, return feature map size. Use index 2 in case of 3D feature maps
                    fm_size = tf.cast(tf.shape(FPN_scale)[2], tf.float32)

                    # Calculate feature stride
                    feature_stride = tf.cast(self.image_size // fm_size, tf.float32)

                    # Base anchor sizes match feature map sizes. Generate anchors at this level
                    anchors = self._generate_anchors(fm_size, base_anchor_size, feature_stride, self.RPN_anchor_ratios, self.RPN_anchor_scales)

                    # Reshape and append to the list
                    anchors = tf.reshape(anchors, [-1, 4])

                    # Copy the anchors along the batches here to mirror what we did with the convolutions
                    batched_anchors = tf.stack([anchors]*self.batch_size, axis=0)
                    anchor_list.append(batched_anchors)

                    # Keep track of which feature pyramid size this set of anchors came from
                    this_FPN_source = tf.ones_like(batched_anchors[:, :, -1:], tf.int16)
                    this_FPN_source *= FPN_scale.get_shape().as_list()[2]
                    FPN_sources.append(this_FPN_source)

                all_level_anchors = tf.concat(anchor_list, axis=1)
                FPN_Source = tf.concat(FPN_sources, axis=1)
                self.anchor_fpn_sources = tf.reshape(FPN_Source, shape=[-1, 1])

            return all_level_anchors

    def _generate_anchors(self, shape, base_anchor_size, feature_stride, ratios, scales, name='generate_anchors'):

        """
        For generating anchors inside the tensorflow computation graph
        :param shape: Spatial shape of the feature map over which to generate anchors
        :param base_anchor_size: The base anchor size for this feature map
        :param feature_stride: int, stride of feature map relative to the image in pixels
        :param ratios: [1D array] of anchor ratios of width/height. i.e [0.5, 1, 2]
        :param scales: [1D array] of anchor scales in original space
        :return:
        """

        # Define a variable scope
        with tf.variable_scope(name):

            # Generate a base anchor
            base_anchor = tf.constant([0, 0, base_anchor_size, base_anchor_size], tf.float32)
            base_anchors = self._enum_ratios(self._enum_scales(base_anchor, scales), ratios)
            _, _, ws, hs = tf.unstack(base_anchors, axis=1)

            # Create sequence of numbers
            x_centers = tf.range(shape, dtype=tf.float32) * feature_stride
            y_centers = tf.range(shape, dtype=tf.float32) * feature_stride

            # Broadcast parameters to a grid of x and y coordinates
            x_centers, y_centers = tf.meshgrid(x_centers, y_centers)
            ws, x_centers = tf.meshgrid(ws, x_centers)
            hs, y_centers = tf.meshgrid(hs, y_centers)

            # Stack anchor centers and box sizes. Reshape to get a list of (x, y) and a list of (h, w)
            anchor_centers = tf.reshape(tf.stack([x_centers, y_centers], 2), [-1, 2])
            box_sizes = tf.reshape(tf.stack([ws, hs], axis=2), [-1, 2])

            # Convert to corner coordinates
            anchors = tf.concat([anchor_centers - 0.5 * box_sizes, anchor_centers + 0.5 * box_sizes], axis=1)

            return anchors

    def _filter_outside_anchors(self, anchors, img_dim):

        """
        Removes anchor proposals with values outside the image
        :param anchors: The anchor proposals [xmin, ymin, xmax, ymax]
        :param img_dim: image dimensions (assumes square input)
        :return: the indices of the anchors not outside the image boundary
        """

        with tf.name_scope('filter_outside_anchors'):

            # Unpack the rank R tensor into multiple rank R-1 tensors along axis
            ymin, xmin, ymax, xmax = tf.unstack(anchors, axis=1)

            # Return True for indices inside the image
            xmin_index, ymin_index = tf.greater_equal(xmin, 0), tf.greater_equal(ymin, 0)
            xmax_index, ymax_index = tf.less_equal(xmax, img_dim), tf.less_equal(ymax, img_dim)

            # Now clean up the indices and return them
            indices = tf.transpose(tf.stack([ymin_index, xmin_index, ymax_index, xmax_index]))
            indices = tf.cast(indices, dtype=tf.int32)
            indices = tf.reduce_sum(indices, axis=1)
            indices = tf.where(tf.equal(indices, tf.shape(anchors)[1]))

            return tf.reshape(indices, [-1, ])

    def _RPN_make_minibatch(self, valid_anchors):

        """
        Takes the valid anchors and generates a minibatch for the RPN with a certain amount of positives and negatives
        :param valid_anchors:
        :return:
        """

        with tf.variable_scope('rpn_minibatch'):

            # Label shape is [N,: ] where 1 is positive, 0 is negative and -1 is ignored.
            # amgtb is [n, 4] where every anchor index is matched with the corresponding highest iou anchor
            labels, anchors_matched_gtboxes, object_mask = self._RPN_process_proposals(valid_anchors)

            # Positive indices are labels = 1.0. Return a reduced size array with these indices as entries
            positive_indices = tf.reshape(tf.where(tf.equal(labels, 1.0)), [-1])
            negative_indices = tf.reshape(tf.where(tf.equal(labels, 0.0)), [-1])

            # Calculate the number of to include (scalar)
            num_positives = tf.minimum(tf.shape(positive_indices)[0], tf.cast(self.RPN_batch_positives_ratio * self.RPN_batch_size, tf.int32))
            num_negatives = tf.minimum(self.RPN_batch_size - num_positives, tf.shape(negative_indices)[0])

            # Retreive a random selection of the positives and negatives
            positive_indices = tf.slice(tf.random_shuffle(positive_indices), begin=[0], size=[num_positives])
            negative_indices = tf.slice(tf.random_shuffle(negative_indices), begin=[0], size=[num_negatives])

            # Join together to create the minibatch indices and randomize
            minibatch_indices = tf.concat([positive_indices, negative_indices], axis=0)
            minibatch_indices = tf.random_shuffle(minibatch_indices)

            # Retreive the ground truth boxes for the indices in the generated minibatch
            minibatch_anchor_matched_gtboxes = tf.gather(anchors_matched_gtboxes, minibatch_indices)

            # Regenerate the labels and object mask for the indices that actually made it to the minibatch
            object_mask = tf.gather(object_mask, minibatch_indices)
            labels = tf.cast(tf.gather(labels, minibatch_indices), tf.int32)

            # Make labels one hot
            labels_one_hot = tf.one_hot(labels, depth=2)

            return minibatch_indices, minibatch_anchor_matched_gtboxes, object_mask, labels_one_hot

    def _RPN_make_minibatch_equalized_prototype(self, valid_anchors):

        """
        Takes the valid anchors and generates a minibatch for the RPN with a certain amount of positives and negatives
        This version makes sure that each image contributes the same amount to the minibatch
        :param valid_anchors: The anchors actually within the image
        :return:
        """

        with tf.variable_scope('rpn_minibatch'):

            # Label shape is [N,: ] where 1 is positive, 0 is negative and -1 is ignored.
            # amgtb is [n, 4] where every anchor index is matched with the corresponding highest iou anchor
            labels, anchors_matched_gtboxes, object_mask = self._RPN_process_proposals(valid_anchors)

            # Reshape to batch sizes
            labels = tf.reshape(labels, [self.batch_size, -1])

            # This approach won't work because the indices are on a per batch basis once you reshape "labels". i.e. you will have duplicate indices
            for b in range(self.batch_size):

                # Positive indices are labels = 1.0. Return a reduced size array with these indices as entries
                positive_indices = tf.reshape(tf.where(tf.equal(labels[b], 1.0)), [-1])
                negative_indices = tf.reshape(tf.where(tf.equal(labels[b], 0.0)), [-1])

                # Calculate the number of to include (scalar)
                num_positives = tf.minimum(tf.shape(positive_indices)[0], tf.cast(self.RPN_batch_positives_ratio * self.RPN_batch_size // self.batch_size, tf.int32))
                num_negatives = tf.minimum(self.RPN_batch_size//self.batch_size - num_positives, tf.shape(negative_indices)[0])

                # Retreive a random selection of the positives and negatives
                positive_indices = tf.slice(tf.random_shuffle(positive_indices), begin=[0], size=[num_positives])
                negative_indices = tf.slice(tf.random_shuffle(negative_indices), begin=[0], size=[num_negatives])

                # Join together to create the minibatch indices and randomize
                minibatch_indices = tf.concat([positive_indices, negative_indices], axis=0)
                minibatch_indices = tf.random_shuffle(minibatch_indices)

            # Retreive the ground truth boxes for the indices in the generated minibatch
            minibatch_anchor_matched_gtboxes = tf.gather(anchors_matched_gtboxes, minibatch_indices)

            # Regenerate the labels and object mask for the indices that actually made it to the minibatch
            object_mask = tf.gather(object_mask, minibatch_indices)
            labels = tf.cast(tf.gather(labels, minibatch_indices), tf.int32)

            # Make labels one hot
            labels_one_hot = tf.one_hot(labels, depth=2)

            return minibatch_indices, minibatch_anchor_matched_gtboxes, object_mask, labels_one_hot

    def _RPN_process_proposals(self, anchors):

        """
        Find positive and negative samples: Assign anchors as object or background
        :param anchors: [valid_anchor_num, 4]
        :return:
                labels, anchors_matched_gtboxes, object_mask
        """

        # Set the variable scope
        with tf.variable_scope('Process_proposals'):

            # At this point, there is one or two GT box per image. and all the anchors are exactly the same for each image
            ious, column = [], 0

            # Calculate the iou for each anchor and gt box gtboxes shape=(8, 1, 5), anchors shape=(186k, 4)
            num_gt_boxes = tf.shape(self.gt_boxes)[0] * tf.shape(self.gt_boxes)[1]
            for z in range (self.batch_size):

                # Use only this batches gt_boxes and anchors
                batch_gtbox = tf.cast(tf.reshape(self.gt_boxes[z, :, :-1], [-1, 4]), tf.float32)
                batch_anchors = tf.reshape(anchors, [self.batch_size, -1, 4])

                # Retreive number of gtboxes for this image and number of anchors
                num_gt = tf.shape(batch_gtbox)[0]

                # Calculate ious and append to the correct portion of the iou array
                batch_iou = self._iou_calculate(batch_anchors[z], batch_gtbox)

                # Pad the ious with zeros along the y axis
                paddings = tf.Variable([[0, 0], [column, (num_gt_boxes - (column + num_gt))]], trainable=False)
                batch_iou = tf.pad(batch_iou, paddings)
                ious.append(batch_iou)

                # Move column start point for next iteration
                column += num_gt

            # Combine anchors.
            iou = tf.concat(ious, axis=0)

            # Saved GT boxes as boxes and the FOREGROUND label as last row. (gt_boxes was saved as 5 columns with last as class, not foreground/bkgrnd)
            gtboxes = tf.cast(tf.reshape(self.gt_boxes[:, :, :-1], [-1, 4]), tf.float32)
            labels = tf.ones(shape=[tf.shape(anchors)[0], ], dtype=tf.float32) * (-1)  # Make all ignored for now

            # Retreive the max iou for each anchor (row) and max iou for each gtbox (column)
            max_iou_each_row = tf.reduce_max(iou, axis=1)
            max_iou_each_column = tf.reduce_max(iou, axis=0)

            # Retreives indices: positives1 and 2 for anchor with iou > 0.7 (default) and anchors with highest IoUs respectively
            positives1 = tf.greater_equal(max_iou_each_row, self.RPN_nms_upper_threshold)  # Any iou index > threshold is True
            positives2 = tf.reduce_sum(tf.cast(tf.equal(iou, max_iou_each_column), tf.float32), axis=1) # Index for any anchor with iou == maxiou
            positives = tf.logical_or(positives1, tf.cast(positives2, tf.bool)) # All anchors that fit the above two conditions            

            # Update labels with positive indices. Pos = 1, ignored and bkgrnd = -1
            labels += 2 * tf.cast(positives, tf.float32)

            # We want to retreive the index of the positive matches (positives is a binary matrix right now)
            matches = tf.cast(tf.argmax(iou, axis=1), tf.int32)     # For each anchor, what column holds the highest iou
            matches = matches * tf.cast(positives, dtype=matches.dtype)  # For each anchor we deem positive, what column holds the highest IOU
            anchors_matched_gtboxes = tf.gather(gtboxes, matches) # For every anchor, return the GT box it matches up with best

            # Anchors matched_gt_boxes is currently returning the GT box of each index. Works fine except for index 0 of which for background should be just 0,
            # Later on the object mask will ignore these anyway

            # Retreive the negatives with iou < 0.3 (default)
            negatives = tf.less(max_iou_each_row, self.RPN_nms_lower_threshold)
            negatives = tf.cast(tf.logical_and(negatives, tf.greater_equal(max_iou_each_row, 0.1)), tf.float32)

            # Update the labels with the negatives: +ive = 1, -ive = 0, ignored = -1
            labels += tf.cast(negatives, tf.float32) # +ive = 1, -ive = 0, ignored = -1
            pos = tf.cast(tf.greater_equal(labels, 1.0), tf.float32) # 0, 1 (pos)
            ignored = tf.cast(tf.equal(labels, -1.0), tf.float32) * (-1) # 0, -1 (ignored)
            labels = pos + ignored

            '''
            Please note that without the abovecorrection, when positive, labels may be >= 1.0: 
            Labels all start at -1
            If all iou < 0.7, the max anchor is set as positive and gets a +2.0 (==1.0)
            If that anchor also has iou < 0.3 it gets another +1.0 (==2.0)
            '''

            # Object mask: 1.0 is object, 0.0 is other
            object_mask = tf.cast(positives, tf.float32)

            # TODO: Glitch returning the same boxes
            return labels, anchors_matched_gtboxes, object_mask

    """
    FRCNN hidden Inside functions
    """

    def _ROI_Align(self):

        """
        Receives non normalized boxes and Performs ROI align using built in tensorflow function
        :param rpn_proposal_boxes: [n, 4] The proposals generated by the RPN
        :param rpn_proposal_scores: [n, ] The objectness scores from the RPN
        :param rpn_proposal_sources: [n, batch#, FPN_dim] the image and feature pyramid source of each proposal
        :return:
        """

        with tf.variable_scope('ROI_Align'):

            # Variables
            roi, src, box = [], [], []
            crop_size = [7, 7]

            # Loop through and generate the rois using bilinear interpolation on a per feature map basis
            # NOTE soource of data augmentation. if y1>y2 the image is flipped vertically, same for x
            for i in range (len(self.conv)):

                # Get valid indices
                dims = self.conv[i].get_shape().as_list()[2]
                vix = tf.where(tf.equal(dims, tf.cast(self.rpn_proposal_sources[:, 1], tf.int32)))

                # Gather all the indices from this FPN
                boxes = tf.gather(self.rpn_proposal_boxes, vix)
                sources = tf.gather(self.rpn_proposal_sources, vix)
                boxes, sources = tf.reshape(boxes, [-1, 4]), tf.reshape(sources, [-1, 2])

                # Normalize the boxes to the input image size.
                rois = boxes / self.image_size

                # Stop gradient propogation to ROI proposals
                rois = tf.stop_gradient(rois)
                sources = tf.stop_gradient(sources)

                # Create all the ROI crops of this FPN dimension
                crop = tf.image.crop_and_resize(self.conv[i], rois, tf.cast(sources[:, 0], tf.int32), crop_size)

                # Save 1: rois (feature map crops), sources (batch, fpn_level) and boxes (anchor box)
                roi.append(crop)
                src.append(sources)
                box.append(boxes)

            # Combine all the crops: TODO: Regenerate batches
            self.FRCNN_rois, self.FRCNN_srcs, self.FRCNN_proposals = tf.concat(roi, axis=0), tf.concat(src, axis=0), tf.concat(box, axis=0)

    def _FRCNN_conv(self):

        """
        Wrapper for the fast RCNN head
        :return:
        """

        with tf.variable_scope('fast_rcnn_net'):

            # Run ROIs through the FCNs, dropout 1st layer if requested
            self.summary = True
            conv = self.fc7_layer('Fc_1', self.FRCNN_rois, self.FC_nodes, True, self.phase_train, BN=True)
            conv = self._linear_layer('Fc_2', conv, self.FC_nodes, phase_train=self.phase_train, BN=True, relu=True, add_bias=True, dim=self.FC_nodes)

            # Run them through the next FC_nodes for class scores
            self.FRCNN_class_logits = self._linear_layer('FCNN_classifier', conv, self.num_classes + 1, dim=self.FC_nodes, phase_train=self.phase_train)

            # Run through the second FC_nodes branch for box scores
            self.FRCNN_box_logits = self._linear_layer('FCNN_regressor', conv, self.num_classes * 4, dim=self.FC_nodes, phase_train=self.phase_train)

    def _FRCNN_process_proposals(self):

        """
        To compare each reference box's label and ground truth box:
        iou >= 0.5 is object, less than that is background
        :param self.rpn_proposal_boxes: [n, 4] outputs of the RPN
        :param self.rpn_proposals_sources: [n, 2] the [batch, Feature_pyramid] source of the reference boxes
        :param self.gt_boxes: [batch, n, 5] with index [-1] being the class label
        :return:
            Each reference boxes matched ground truth box
            object mask indicating 1 for object, 0 for background
            category label indicating a box's class with one hot encoding [n, num_classes + 1]
        """

        with tf.variable_scope('FRCNN_Process_proposals'):

            # There is one or two GT box per image.
            ious, sources, column = [], [], 0

            # Calculate the iou for each proposal and gt box gtboxes shape=(8, 1, 5), proposals shape=(256, 4)
            num_gt_boxes = tf.shape(self.gt_boxes)[0] * tf.shape(self.gt_boxes)[1]
            for z in range (self.batch_size):

                # Gather the proposals in this batch
                indices = tf.where(tf.equal(z, tf.cast(self.rpn_proposal_sources[:, 0], tf.int32)))
                batch_gtbox = tf.cast(tf.reshape(self.gt_boxes[z, :, :-1], [-1, 4]), tf.float32)
                batch_proposals = tf.reshape(tf.gather(self.rpn_proposal_boxes, indices), [-1, 4])
                batch_sources = tf.reshape(tf.gather(self.rpn_proposal_sources, indices), [-1, 2])

                # Retreive number of gtboxes for this image
                num_gt = tf.shape(batch_gtbox)[0]

                # Calculate ious and append to the correct portion of the iou array
                batch_iou = self._iou_calculate(batch_proposals, batch_gtbox)

                # Pad the ious with zeros along the y axis?
                paddings = tf.Variable([[0, 0], [column, (num_gt_boxes - (column + num_gt))]], trainable=False)
                batch_iou = tf.pad(batch_iou, paddings)
                ious.append(batch_iou)

                # Keep track of sources too
                sources.append(batch_sources)

                # Move column start point for next iteration
                column += num_gt

            # Combine anchors.
            iou = tf.concat(ious, axis=0)
            source = tf.concat(sources, 0)

            # Calculate matches. Again if theres no iou, index 0 is used as a max but will be nulled by the object mask
            matches = tf.cast(tf.argmax(iou, axis=1), tf.int32)
            max_iou_each_row = tf.reduce_max(iou, axis=1)

            # Saved GT boxes. gt_boxes was saved as 5 columns with last column referring to the class
            gtboxes = tf.cast(tf.reshape(self.gt_boxes[:, :, :-1], [-1, 4]), tf.float32)
            gtlabels = tf.cast(tf.reshape(self.gt_boxes[:, :, -1], [-1, 1]), tf.float32)

            # Retreive positive matches (1 = match) and any matched GTBoxes (again 0 = 0 or no match but will be accounnted for in mask)
            positives = tf.cast(tf.greater_equal(max_iou_each_row, self.FRCNN_positives_threshold), tf.float32)
            reference_matched_gtboxes = tf.gather(gtboxes, matches)

            # Generate the object mask to prevent gradient calculation on background boxes
            object_mask = tf.cast(positives, tf.float32)

            # Retreive labels
            label = tf.gather(gtlabels, matches)
            label = tf.squeeze(label) * tf.squeeze(positives)

            return reference_matched_gtboxes, object_mask, label

    def _FRCNN_make_minibatch(self):

        """
        Makes a minibatch to help us calculate loss in the FRCNN
        :return:
        """

        with tf.variable_scope('FRCNN_Minibatch'):

            # Retreive the  processed proposals with corresponding object mask and labels
            gtmatches, object_mask, labels = self._FRCNN_process_proposals()

            # Positive indices this time are anywhere the object mask is not 0. The reverse is true too
            positive_indices = tf.reshape(tf.where(tf.not_equal(object_mask, 0.0)), [-1])
            negative_indices = tf.reshape(tf.where(tf.equal(object_mask, 0.0)), [-1])

            # Calculate the number of to include (scalar)
            num_positives = tf.minimum(tf.shape(positive_indices)[0], tf.cast(self.FRCNN_batch_positives_ratio * self.FRCNN_batch_size, tf.int32))
            num_negatives = tf.minimum(self.RPN_batch_size - num_positives, tf.shape(negative_indices)[0])

            # Retreive a random selection of the positives and negatives
            positive_indices = tf.slice(tf.random_shuffle(positive_indices), begin=[0], size=[num_positives])
            negative_indices = tf.slice(tf.random_shuffle(negative_indices), begin=[0], size=[num_negatives])

            """
            ROI Visualizations
            """
            # Generate positive rois (feature map crops), sources [batch, FPN] and proposals (anchors)
            positive_rois = tf.gather(self.FRCNN_rois, positive_indices)
            positive_sources = tf.gather(self.FRCNN_srcs, positive_indices)
            positive_proposals = tf.gather(self.FRCNN_proposals, positive_indices)
            img_shape = tf.cast(self.image_size, tf.float32)

            # Join together to create the minibatch indices and randomize
            minibatch_indices = tf.concat([positive_indices, negative_indices], axis=0)
            minibatch_indices = tf.random_shuffle(minibatch_indices)

            # Retreive the ground truth boxes for the indices in the generated minibatch
            minibatch_gtboxes = tf.gather(gtmatches, minibatch_indices)

            # Regenerate the labels and object mask for the indices that actually made it to the minibatch
            object_mask = tf.gather(object_mask, minibatch_indices)
            labels = tf.cast(tf.gather(labels, minibatch_indices), tf.int32)

            # Make labels one hot
            labels_one_hot = tf.one_hot(labels, depth=self.num_classes + 1)

            return minibatch_indices, minibatch_gtboxes, object_mask, labels_one_hot

    def _FRCNN_loss(self):

        """
        Generates the loss for the faster RCNN head
        :return:
        """

        with tf.variable_scope('faster_rcnn_loss'):

            # Retreive the last 3 functions basically
            batch_indices, batch_gtboxes, batch_object_mask, batch_lbl_one_hot = self._FRCNN_make_minibatch()

            # Get the anchor box proposals, box logits, class logits and sources for the batch indices
            minibatch_reference_boxes = tf.gather(self.FRCNN_proposals, batch_indices)
            minibatch_box_logits = tf.gather(self.FRCNN_box_logits, batch_indices)
            minibatch_class_logits = tf.gather(self.FRCNN_class_logits, batch_indices)
            minibatch_sources = tf.gather(self.FRCNN_srcs, batch_indices)

            # Draw boxes
            proposal_images = self.draw_box_in_img_batch(self.Image_batch, minibatch_reference_boxes, minibatch_sources[:, 0])
            for z in range (len(proposal_images)): tf.summary.image('/positive_proposals', proposal_images[z], max_outputs=self.batch_size)

            # Retreive the box deltas
            gtbox_deltas = self._find_deltas(batch_gtboxes, minibatch_reference_boxes)
            gtbox_deltas = tf.tile(gtbox_deltas, [1, self.num_classes])

            class_weights_list = []
            batch_lbl_one_hot = tf.reshape(batch_lbl_one_hot, [-1, self.num_classes + 1])
            category_list = tf.unstack(batch_lbl_one_hot, axis=1)

            for i in range(1, self.num_classes + 1):
                tmp_class_weights = tf.ones(shape=[tf.shape(gtbox_deltas)[0], 4], dtype=tf.float32)
                tmp_class_weights = tmp_class_weights * tf.expand_dims(category_list[i], axis=1)
                class_weights_list.append(tmp_class_weights)
            class_weights = tf.concat(class_weights_list, axis=1)  # [minibatch_size, num_classes*4]

            # loss
            with tf.variable_scope('frcnn_classification_loss'):

                # Calculate the softmax class loss
                fast_rcnn_classification_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=batch_lbl_one_hot, logits=minibatch_class_logits)
                fast_rcnn_classification_loss = tf.reduce_mean(fast_rcnn_classification_loss)
                fast_rcnn_classification_loss *= self.FRCNN_class_loss_weight
                tf.add_to_collection('losses', fast_rcnn_classification_loss)

            with tf.variable_scope('frcnn_location_loss'):

                # Calculate the smooth L1 loss for the boxes
                fast_rcnn_location_loss = self._smooth_l1_loss(predicted_boxes=minibatch_box_logits, gtboxes=gtbox_deltas, object_weights=batch_object_mask, classes_weights=class_weights)
                fast_rcnn_location_loss *= self.FRCNN_box_loss_weight
                tf.add_to_collection('losses', fast_rcnn_location_loss)

            self.FRCNN_location_loss = fast_rcnn_location_loss
            self.FRCNN_classification_loss = fast_rcnn_classification_loss

    def _FRCNN_proposals_only(self):

        """
        Utility function for running a forward pass only without loss calculation and performing NMS
        :return:
        """

        with tf.variable_scope('FRCNN_Proposals_only'):

            return 1, 2, 3, 4


    """
    Loss functions
    """

    def _smooth_l1_loss(self, predicted_boxes, gtboxes, object_weights, classes_weights=None):

        """
        TODO: Calculates the smooth L1 losses
        :param predicted_boxes: The filtered anchors
        :param gtboxes:ground truth boxes
        :param object_weights: The mask map indicating whether this is an object or not
        :return:
        """

        diff = predicted_boxes - gtboxes
        absolute_diff = tf.cast(tf.abs(diff), tf.float32)

        if classes_weights is None:

            anchorwise_smooth_l1norm = tf.reduce_sum(tf.where(tf.less(absolute_diff, 1),
                                    0.5 * tf.square(absolute_diff), absolute_diff - 0.5), axis=1) * object_weights

        else:

            anchorwise_smooth_l1norm = tf.reduce_sum(tf.where(tf.less(absolute_diff, 1),
                                        0.5 * tf.square(absolute_diff)*classes_weights, (absolute_diff - 0.5)*classes_weights), axis=1) * object_weights

        return tf.reduce_mean(anchorwise_smooth_l1norm, axis=0)

    """
    Bounding box functions
    """

    def _iou_calculate(self, boxes1, boxes2):

        """
        Calculates the IOU of two boxes
        :param boxes1: [n, 4] [ymin, xmin, ymax, xmax]
        :param boxes2: [n, 4]
        :return: Overlaps of each box pair (aka DICE score)
        """

        # Just a name scope this time
        with tf.name_scope('iou_calculate'):

            # Split the coordinates
            ymin_1, xmin_1, ymax_1, xmax_1 = tf.split(boxes1, 4, axis=1)  # ymin_1 shape is [N, 1]..
            ymin_2, xmin_2, ymax_2, xmax_2 = tf.unstack(boxes2, axis=1)  # ymin_2 shape is [M, ]..

            # Retreive any overlaps of the corner points of the box
            max_xmin, max_ymin = tf.maximum(xmin_1, xmin_2), tf.maximum(ymin_1, ymin_2)
            min_xmax, min_ymax = tf.minimum(xmax_1, xmax_2), tf.minimum(ymax_1, ymax_2)

            # Retreive overlap along each dimension: Basically if the upper right corner of one box is above the lower left of another, there is overlap
            overlap_h = tf.maximum(0., min_ymax - max_ymin)  # avoid h < 0
            overlap_w = tf.maximum(0., min_xmax - max_xmin)

            # Cannot overlap if one of the dimension overlaps is 0
            overlaps = overlap_h * overlap_w

            # Calculate the area of each box
            area_1 = (xmax_1 - xmin_1) * (ymax_1 - ymin_1)  # [N, 1]
            area_2 = (xmax_2 - xmin_2) * (ymax_2 - ymin_2)  # [M, ]

            # Calculate overlap (intersection) over union like Dice score. Union is just the areas added minus the overlap
            iou = overlaps / (area_1 + area_2 - overlaps)

            return iou

    def calculate_overlaps_mask_box(self, mask, box):

        """
        Calculates the overlap between a mask and a box:
        :param mask: 2D mask
        :param box: [y1, x1, y2, x2]
        :return: Number of nonzero mask pixels / box area
        """

        # Calculate the nonzero pixels of the mask in the range of the box
        nonzero = np.count_nonzero(mask[box[0]:box[2], box[1]:box[3]])

        # Calculate the area of the box
        area = (box[2], - box[0]) * (box[3] - box[1])

        return nonzero / area

    def calculate_overlaps_masks(self, masks1, masks2):

        """
        Calculates the IoU overlaps between two sets of masks
        :param masks1: [height, width, instances]
        :param masks2: [height, width, instances]
        :return:
        """

        # If either set of masks is empty return empty result
        if masks1.shape[0] == 0 or masks2.shape[0] == 0: return np.zeros((masks1.shape[0], masks2.shape[-1]))

        # flatten masks and compute their areas
        masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
        masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
        area1, area2 = np.sum(masks1, axis=0), np.sum(masks2, axis=0)

        # intersections and union
        intersections = np.dot(masks1.T, masks2)
        union = area1[:, None] + area2[None, :] - intersections

        return intersections / union

    def calculate_non_max_suppression(self, boxes, scores, IoU_threshold):

        """
        Groups highly overlapped boxes for the same class and selects the most confident predictions only.
        This avoids duplicates for the same object
        :param boxes: [N, (y1, x1, y2, x2)]
        :param scores: 1-D array of the box scores
        :param IoU_threshold: float, IoU threshold to use for filtering
        :return: List of the final boxes
        """

        # Debug code first for box shape then making sure boxes are floats
        assert boxes.shape[0] > 0, "No boxes passed to non max suppression"
        if boxes.dtype.kind != 'f': boxes = boxes.astype(np.float32)

        # Calculate box areas
        y1, x1, y2, x2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        area = (y2 - y1) * (x2 - x1)

        # Retreive box indices sorted by higest scores first
        indices = scores.argsort()[:, :, -1]
        picks = []

        while len(indices) > 0:
            # Select the top box and add it's index to the list
            i = indices[0]
            picks.append(i)

            # Compute IoU of the picked box with the rest of the boxes
            iou = self.calculate_iou(boxes[i], boxes[indices[1:]], area[i], area[indices[1:]])

            # Find boxes with IoU over threshold and return indices. Add 1 to get indices into [indices]
            remove_indices = np.where(iou > IoU_threshold)[0] + 1

            # Remove the indices of the overlapped boxes and the picked box
            indices = np.delete(indices, remove_indices)
            indices = np.delete(indices, 0)

        return np.array(picks, dtype=np.int32)

    """
    Testing functions: TODO: Move to the SODtester class
    """

    def trim_zeros(self, x):

        """
        Remove all rows of a tensor that are all zeros
        :param x: [rows, columns]
        :return:
        """

        assert len(x.shape == 2)
        return x[~np.all(x == 0, axis=1)]

    def test_calculate_matches(self, gt_boxes, gt_class_ids, gt_masks, pred_boxes, pred_class_ids, pred_scores, pred_masks, iou_threshold=0.5, score_threshold=0.0):

        """
        Finds matches between the predictions and ground truth
        :param gt_boxes:
        :param gt_class_ids:
        :param gt_masks:
        :param pred_boxes:
        :param pred_class_ids:
        :param pred_scores
        :param pred_masks:
        :param iou_threshold:
        :param score_threshold:
        :return:
                gt_match: 1-D array. For each GT box it has the index of the matched predicted box.
                pred_match: 1-D array. For each predicted box, it has the index of the matched ground truth box.
                overlaps: [pred_boxes, gt_boxes] IoU overlaps.
        """

        # Trim zero padding of the masks
        gt_boxes, pred_boxes = self.trim_zeros(gt_boxes), self.trim_zeros(pred_boxes)
        gt_masks, pred_scores = gt_masks[..., :gt_boxes.shape[0]], pred_scores[:pred_boxes.shape[0]]

        # Sort predictions by scores from high to low
        indices = np.argsort(pred_scores)[::-1]
        pred_boxes, pred_class_ids = pred_boxes[indices], pred_class_ids[indices]
        pred_scores, pred_masks = pred_scores[indices], pred_masks[..., indices]

        # Compute iou overlaps
        overlaps = self.calculate_overlaps_masks(pred_masks, gt_masks)

        # Loop through predictions and find the matching ground truth boxes
        match_count = 0
        pred_match = -1 * np.ones([pred_boxes.shape[0]])
        gt_match = -1 * np.ones([gt_boxes.shape[0]])
        for i in range(len(pred_boxes)):

            # Find the best best matching ground truth box: 1. sort matches by overlap score
            sorted_index = np.argsort(overlaps[i])[::-1]

            # Remove low scores
            low_score_index = np.where(overlaps[i, sorted_index] < score_threshold)[0]
            if low_score_index.size > 0: sorted_index = sorted_index[:low_score_index[0]]

            # Find the match
            for j in sorted_index:

                # if ground truth box is already matched, move on
                if gt_match[j] > 0: continue

                # If IoU is smaller than the threshold, end the loop
                iou = overlaps[i, j]
                if iou < iou_threshold: break

                # Check for a match
                if pred_class_ids[i] == gt_class_ids[j]:
                    match_count += 1
                    gt_match[j], pred_match[i] = i, j
                    break

        return gt_match, pred_match, overlaps

    def test_calculate_PPV(self, gt_boxes, gt_class_ids, gt_masks, pred_boxes, pred_class_ids, pred_scores, pred_masks, iou_threshold=0.5):

        """
        Calculates the average precision (AKA PPV) at a set IoU threshold and Sensitivity (AKA Recall)
        :param gt_boxes:
        :param gt_class_ids:
        :param gt_masks:
        :param pred_boxes:
        :param pred_class_ids:
        :param pred_scores:
        :param pred_masks:
        :param iou_threshold:
        :return:
                mAP: Mean Average Precision
                precisions: List of precisions at different class score thresholds.
                recalls: List of recall values at different class score thresholds.
                overlaps: [pred_boxes, gt_boxes] IoU overlaps.
        """

        # Get matches and overlaps
        gt_match, pred_match, overlaps = self.test_calculate_matches(gt_boxes, gt_class_ids, gt_masks, pred_boxes, pred_class_ids, pred_scores, pred_masks, iou_threshold)

        # Compute precision and recall at each prediction box step
        precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
        recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

        # Ensure precision values increase but don't decrease. To ensure the PPV at each sensitivity threshold is the max for all following thresholds
        for i in range(len(precisions) - 2, -1, -1): precisions[i] = np.maximum(precisions[i], precisions[i + 1])

        # Compute mean average precision over the sensitivity range
        idx = np.where(recalls[:-1] != recalls[1:])[0] + 1
        mAP = np.sum((recalls[idx] - recalls[idx - 1]) * precisions[idx])

        return mAP, precisions, recalls, overlaps

    def test_calculate_PPV_range(self, gt_box, gt_class_id, gt_mask, pred_box, pred_class_id, pred_score, pred_mask, iou_thresholds=0.5, verbose=1):

        """
        Calculates the average precision (PPV) over a range or IoU thresholds.
        :param gt_box:
        :param gt_class_id:
        :param gt_mask:
        :param pred_boxe:
        :param pred_class_id:
        :param pred_score:
        :param pred_mask:
        :param iou_threshold:
        :param verbose:
        :return:
        """

        # Default is 0.5 to 0.95 with increments of 0.05
        iou_thresholds = iou_thresholds or np.arange(0.5, 0.95, 0.05)

        # compute PPV over the thresholds
        PPV = []
        for iou_threshold in iou_thresholds:

            ppv, precisions, recals, overlaps = self.test_calculate_PPV(gt_box, gt_class_id, gt_mask, pred_box, pred_class_id, pred_score, pred_mask, iou_threshold=iou_threshold)
            PPV.append(ppv)

            # Display information if desired
            if verbose: print('PPV: %.2f: \t %.3f' % (iou_threshold, ppv))

        PPV = np.array(PPV).mean()
        if verbose: print('PPV: %.2f: \t %.3f: \t %.3f' % (iou_thresholds[0], iou_thresholds[-1], PPV))

        return PPV

    def test_calculate_SN(self, pred_boxes, gt_boxes, iou):

        """
        Calculates the recall (AKA Sensitivity) at the given IoU threshold
        :param pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
        :param gt_boxes: [N, (y1, x1, y2, x2)]
        :param iou: threshold iou
        :return:
        """

        # Measure overlaps
        overlaps = self.calculate_overlaps_IoU_boxes(pred_boxes, gt_boxes)
        iou_max, iou_argmax = np.max(overlaps, axis=1), np.argmax(overlaps, axis=1)
        positive_ids = np.where(iou_max >= iou)[0]
        matched_gt_boxes = iou_argmax[positive_ids]

        # Calculate sensitivity
        recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]

        return recall, positive_ids

    """
    Display functoins to show the boxes being made
    """

    def draw_box_in_img_batch(self, img_batch, boxes, box_batch):

        """
        Draws a bounding box in the image batch
        :param img_batch: the images including the batch number [batch, z, y, x, c]
        :param boxes: input boxes without the batch dimension [n, 4]
        :param box_batch: The batches index corresponding to each input box [n]
        :return:
        """

        # Normalize the boxes
        boxes = tf.cast(boxes, tf.float32)
        img_dims = tf.shape(img_batch)[2]
        boxes = tf.div(boxes, tf.cast(img_dims, tf.float32))

        # Arrange by batches: Get indices per, gather indices, concat together
        batched_boxes, pad_before = [], 0
        for z in range (self.batch_size):

            # Gather the indices of boxes for this batch
            indices = tf.where(tf.equal(z, tf.cast(box_batch, tf.int32)))
            batch_boxes = tf.reshape(tf.gather(boxes, tf.squeeze(indices)), [-1, 4])

            # Expand dims
            batch_boxes = tf.expand_dims(batch_boxes, 0)

            # IPad the rest of the batches with zeros
            pad_after = self.batch_size - (z + 1)
            paddings = tf.Variable([[pad_before, pad_after], [0, 0], [0,0]], trainable=False)
            batch_boxes = tf.pad(batch_boxes, paddings)

            # Return a copy of the images with the bounding boxes drawn. Use the middle index if 3D:
            if len(img_batch.get_shape().as_list()) == 5:
                boxed_images = tf.image.draw_bounding_boxes(tf.image.grayscale_to_rgb(img_batch[:, 2, ::]), batch_boxes)
            else:
                boxed_images = tf.image.draw_bounding_boxes(tf.image.grayscale_to_rgb(img_batch), batch_boxes)

            # Append
            batched_boxes.append(boxed_images)
            pad_before +=1

        return batched_boxes


    def draw_colored_box(self, img_batch, boxes, text, box_batch):

        """
        Draws boxes with color using open CV
        :param img_batch: [batch, z, y, x, c]
        :param boxes: [n, 4]
        :param text: The text to put in the box
        :param box_batch: [batch] What image batch each box index belongs to
        :return:
        """

        batch_images_with_boxes = []

        # Define a draw box function with open CV to later tensorflow-rize
        def draw_box_cv(img, boxes, text):

            # Add a grayscale value to the image TODO: Was in color
            img = img + np.array([116.779])
            boxes = boxes.astype(np.int64)

            # Normalize image
            img = np.array(img * 255 / np.max(img), np.uint8)

            # Loop through boxes we need to draw
            for box in boxes:

                # Retreive coordinates and a random color
                ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
                color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))

                # Generate a rectangle
                cv2.rectangle(img, pt1=(xmin, ymin), pt2=(xmax, ymax), color=color, thickness=2)

            # Generate overlay text
            text = str(text)
            cv2.putText(img , text=text, orig=((img.shape[1]) // 2, (img.shape[0]) //2), fontFace=3, fontScale=1, color=(255, 0, 0))

            #
            img = img[:, :, -1::-1]

            return img

        # Split the image batch into a list of individual image tensors
        img_singles = tf.unstack(img_batch)

        for z in range (len(img_singles)):

            # Gather indices of the boxes for this specific image in the batch
            box_indices = tf.reshape(tf.where(tf.equal(box_batch, z)), [-1])
            img_boxes = tf.gather(boxes, box_indices)

            # Generate the image tensor
            img_tensor = tf.squeeze(img_singles[z], 0)
            img_tensor_with_boxes = tf.py_func(draw_box_cv, inp=[img_tensor, img_boxes, text], Tout=[tf.uint8])

            # Reshape and append
            img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
            batch_images_with_boxes.append(img_tensor_with_boxes)

        return tf.concat(batch_images_with_boxes, axis=1)


    """
    Hidden Utility Functions
    """

    def _enum_scales(self, base_anchor, anchor_scales, name='enum_scales'):

        '''
        :param base_anchor: [y_center, x_center, h, w]
        :param anchor_scales: different scales, like [0.5, 1., 2.0]
        :return: return base anchors in different scales.
                Example:[[0, 0, 128, 128],[0, 0, 256, 256],[0, 0, 512, 512]]
        '''
        with tf.variable_scope(name):
            anchor_scales = base_anchor * tf.constant(anchor_scales, dtype=tf.float32, shape=(len(anchor_scales), 1))
            return anchor_scales

    def _enum_ratios(self, anchors, anchor_ratios, name='enum_ratios'):

        '''
        :param anchors: base anchors in different scales
        :param anchor_ratios:  ratio = h / w
        :return: base anchors in different scales and ratios
        '''

        # TODO: May not be necessary to declare variable scope here (already 3 parent scopes)
        with tf.variable_scope(name):
            # Unstack along the vertical dimension
            _, _, hs, ws = tf.unstack(anchors, axis=1)

            # Calculate squares of the anchor ratios
            sqrt_ratios = tf.sqrt(anchor_ratios)
            sqrt_ratios = tf.expand_dims(sqrt_ratios, axis=1)

            # Reshape the anchors
            ws = tf.reshape(ws / sqrt_ratios, [-1])
            hs = tf.reshape(hs * sqrt_ratios, [-1])
            # assert tf.shape(ws) == tf.shape(hs), 'h shape is not equal w shape'

            num_anchors_per_location = tf.shape(ws)[0]

            return tf.transpose(tf.stack([tf.zeros([num_anchors_per_location, ]), tf.zeros([num_anchors_per_location, ]), ws, hs]))

    def _find_deltas(self, boxes, anchors, scale_factors=None):

        """
        Generate deltas to transform source offsets into destination anchors
        :param boxes: BoxList holding N boxes to be encoded.
        :param anchors: BoxList of anchors.
        :param: scale_factors: scales location targets when using joint training
        :return:
          a tensor representing N anchor-encoded boxes of the format
          [ty, tx, th, tw].
        """

        # Convert anchors to the center coordinate representation.
        ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
        ymia, xmia, ymaa, xmaa = tf.unstack(anchors, axis=1)
        cenx = (xmin + xmax) / 2
        cenxa = cenxa = (xmia + xmaa) / 2
        ceny = (ymin + ymax) / 2
        cenya = (ymia + ymaa) / 2
        w = xmax - xmin
        h = ymax - ymin
        wa = xmaa - xmia
        ha = ymaa - ymia

        # Avoid NaN in division and log below.
        ha += 1e-8
        wa += 1e-8
        h += 1e-8
        w += 1e-8

        # Calculate the normalized translations required
        tx = (cenx - cenxa) / wa
        ty = (ceny - cenya) / ha
        tw = tf.log(w / wa)
        th = tf.log(h / ha)

        # Scales location targets as used in paper for joint training.
        if scale_factors:
            ty *= scale_factors[0]
            tx *= scale_factors[1]
            th *= scale_factors[2]
            tw *= scale_factors[3]

        return tf.transpose(tf.stack([ty, tx, th, tw]))

    def _apply_deltas(self, rel_codes, anchors, scale_factors=None):

        """
        Applies the delta offsets to the anchors to find
        :param rel_codes: a tensor representing N anchor-encoded boxes.
        :param anchors: BoxList of anchors.
        :param scale_factors:
        :return: boxes: BoxList holding N bounding boxes.
        """

        # Convert anchors to the center coordinate representation.
        ymia, xmia, ymaa, xmaa = tf.unstack(anchors, axis=1)
        cenxa = cenxa = (xmia + xmaa) / 2
        cenya = (ymia + ymaa) / 2
        wa = xmaa - xmia
        ha = ymaa - ymia

        ty, tx, th, tw = tf.unstack(rel_codes, axis=1)

        if scale_factors:
            ty /= scale_factors[0]
            tx /= scale_factors[1]
            th /= scale_factors[2]
            tw /= scale_factors[3]

        w = tf.exp(tw) * wa
        h = tf.exp(th) * ha
        ycenter = ty * ha + cenya
        xcenter = tx * wa + cenxa

        ymin = ycenter - h / 2.
        xmin = xcenter - w / 2.
        ymax = ycenter + h / 2.
        xmax = xcenter + w / 2.

        return tf.transpose(tf.stack([ymin, xmin, ymax, xmax]))

    # Overwrite the convolution function wrapper to include the reuse flag for our RPN to work on a FPN output. Remove downsample and dropout
    def _convolution_RPN(self, scope, X, F, K, S=1, padding='SAME', phase_train=None, BN=True, relu=True, bias=True, reuse=None, summary=True):

        """
        This is a wrapper for convolutions
        :param scope:
        :param X: Output of the prior layer
        :param F: Convolutional filter size
        :param K: Number of feature maps
        :param S: Stride
        :param padding: 'SAME' or 'VALID'
        :param phase_train: For batch norm implementation
        :param BN: whether to perform batch normalization
        :param relu: bool, whether to do the activation function at the end
        :param bias: whether to include a bias term
        :param reuse: wehther we will be reusing this layer
        :param summary: whether to output a summary
        :return:
        """

        # Set channel size based on input depth
        C = X.get_shape().as_list()[-1]
        B = X.get_shape().as_list()[0]

        # Set the scope
        with tf.variable_scope(scope, reuse=reuse) as scope:

            # Set training phase variable
            self.training_phase = phase_train

            # Define the Kernel. Can use Xavier init: contrib.layers.xavier_initializer())
            kernel = tf.get_variable('Weights', shape=[F, F, C, K], initializer=tf.contrib.layers.variance_scaling_initializer())

            # Add to the weights collection
            tf.add_to_collection('weights', kernel)

            # Perform the actual convolution
            conv = tf.nn.conv2d(X, kernel, [1, S, S, 1], padding=padding)

            # Add in the bias
            if bias:
                bias = tf.get_variable('Bias', shape=[K], initializer=tf.constant_initializer(0.0))
                tf.add_to_collection('biases', bias)
                conv = tf.nn.bias_add(conv, bias)

            # Relu activation
            if relu: conv = tf.nn.relu(conv, name=scope.name)

            # Apply the batch normalization. Updates weights during training phase only
            if BN: conv = self.batch_normalization(conv, phase_train, scope)

            # Create a histogram/scalar summary of the conv1 layer
            if summary: self._activation_summary(conv)

            return conv

    def _linear_layer(self, scope, X, neurons, dropout=False, phase_train=True, keep_prob=0.5, BN=False, relu=False, add_bias=False, dim=16):
        """
        Wrapper for implementing a linear layer without or without relu/bn/dropout
        :param scope: internal name
        :param X: input of prior layer
        :param neurons: desired connection number
        :param dropout: whether to use dropout
        :param phase_train: are we in train or test phase
        :param keep_prob: if using dropout, the keep prob
        :param BN: whether to use batch norm
        :param relu: whether to use a nonlinearity
        :param add_bias: Whether to add a bias term here
        :param dim: Dimensions in neurons of the prior layer
        :return:
        """

        # The linear layer Dimensions:
        with tf.variable_scope(scope) as scope:

            # Initialize the weights
            weights = tf.get_variable('weights', shape=[dim, neurons], initializer=tf.contrib.layers.variance_scaling_initializer())

            # Add to the collection of weights
            tf.add_to_collection('weights', weights)

            # Do the math
            linear = tf.matmul(X, weights)

            # Batch norm without biases
            if BN: linear = self.batch_normalization(linear, phase_train, 'LinearNorm')

            # add biases
            if add_bias:

                # Define the biases
                bias = tf.get_variable('Bias', shape=[neurons], initializer=tf.constant_initializer(0.0))

                # Add to the weights collection
                tf.add_to_collection('biases', bias)

                # Apply the biases
                linear = tf.nn.bias_add(linear, bias)

            # relu for nonlinear linear layers. no relu for linear regressions
            if relu: linear = tf.nn.relu(linear, name=scope.name)

            # Dropout here if wanted and in train phase
            if (phase_train == True) and dropout: linear = tf.nn.dropout(linear, keep_prob)

            # Activation summary
            if self.summary: self._activation_summary(linear)

            return linear

    def _clip_boxes_to_img_boundaries(self, decode_boxes, img_shape):

        '''
        For testing: Whe the RPN adjusts the anchor boxes to lie outside the image boundaries, clip them shits back in
        :param decode_boxes: The boxes to clip, adjusted RPN outputs
        :param img_shape: the dimension of the image
        :return: decode boxes
        '''

        with tf.name_scope('clip_boxes_to_img_boundaries'):

            ymin, xmin, ymax, xmax = tf.unstack(decode_boxes, axis=1)

            xmin = tf.maximum(xmin, 0.0)
            xmin = tf.minimum(xmin, tf.cast(img_shape, tf.float32))

            ymin = tf.maximum(ymin, 0.0)
            ymin = tf.minimum(ymin, tf.cast(img_shape, tf.float32))  # avoid xmin > img_w, ymin > img_h

            xmax = tf.minimum(xmax, tf.cast(img_shape, tf.float32))
            ymax = tf.minimum(ymax, tf.cast(img_shape, tf.float32))

            return tf.transpose(tf.stack([ymin, xmin, ymax, xmax]))

    def _non_max_suppression(self, boxes, scores, iou_threshold, max_output_size, name='non_maximal_suppression'):

        """
        Tensorflow has a function for us!
        :param boxes:
        :param scores:
        :param iou_threshold:
        :param max_output_size:
        :param name:
        :return:
        """

        with tf.variable_scope(name):

            nms_index = tf.image.non_max_suppression(boxes=boxes, scores=scores, max_output_size=max_output_size, iou_threshold=iou_threshold, name=name )
            return nms_index

    def _pad_boxes_zeros(self, boxes, scores, sources, max_num_of_boxes):

        '''
        if num of boxes is less than max num of boxes, we need to pad with zeros[0, 0, 0, 0]
        :param boxes:
        :param scores: [-1]
        :param sources:
        :param max_num_of_boxes:
        :return:
        '''

        # Figure out how much to pad
        pad_num = tf.cast(max_num_of_boxes, tf.int32) - tf.shape(boxes)[0]

        # Create a zero pad array
        zero_boxes = tf.zeros(shape=[pad_num, 4], dtype=boxes.dtype)
        zero_scores = tf.zeros(shape=[pad_num], dtype=scores.dtype)
        zero_sources = tf.zeros(shape=[pad_num, 2], dtype=sources.dtype)

        # Concat the arrays
        final_boxes = tf.concat([boxes, zero_boxes], axis=0)
        final_scores = tf.concat([scores, zero_scores], axis=0)
        final_sources = tf.concat([sources, zero_sources], axis=0)

        return final_boxes, final_scores, final_sources

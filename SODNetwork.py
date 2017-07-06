"""
SOD Network contains function wrappers for various tensorflow tasks including:

convolutions, deconvolutions, 3D convolutions, 

"""

import tensorflow as tf
import numpy as np


class SODMatrix():

    """
    SOD Loader class is a class for loading all types of data into protocol buffers
    """

    def __init__(self):

        pass


    """
     Convolution wrappers
    """

    def convolution(self, scope, X, F, K, S=2, padding='SAME', phase_train=None, summary=True, BN=True, relu=True):
        """
        This is a wrapper for convolutions
        :param scope:
        :param X: Output of the prior layer
        :param F: Convolutional filter size
        :param K: Number of feature maps
        :param S: Stride
        :param padding: 'SAME' or 'VALID'
        :param phase_train: For batch norm implementation
        :param summary: whether to produce a tensorboard summary of this layer
        :param BN: whether to perform batch normalization
        :param relu: bool, whether to do the activation function at the end
        :return:
        """

        # Set channel size based on input depth
        C = X.get_shape().as_list()[3]

        # Set the scope
        with tf.variable_scope(scope) as scope:

            # Define the Kernel. Can use Xavier init: contrib.layers.xavier_initializer())
            kernel = tf.get_variable('Weights', shape=[F, F, C, K],
                                     initializer=tf.contrib.layers.xavier_initializer())

            # Add to the weights collection
            tf.add_to_collection('weights', kernel)

            # Perform the actual convolution
            conv = tf.nn.conv2d(X, kernel, [1, S, S, 1], padding=padding)  # Create a 2D tensor with BATCH_SIZE rows

            # Apply the batch normalization. Updates weights during training phase only
            if BN:
                conv = tf.cond(phase_train,
                               lambda: tf.contrib.layers.batch_norm(conv, activation_fn=None, center=True, scale=True,
                                                            updates_collections=None, is_training=True, reuse=None,
                                                            scope=scope, decay=0.9, epsilon=1e-5),
                               lambda: tf.contrib.layers.batch_norm(conv, activation_fn=None, center=True, scale=True,
                                                            updates_collections=None, is_training=False, reuse=True,
                                                            scope=scope, decay=0.9, epsilon=1e-5))

            # Relu activation
            if relu: conv = tf.nn.relu(conv, name=scope.name)

            # Create a histogram/scalar summary of the conv1 layer
            if summary: self._activation_summary(conv)

            return conv


    def convolution_3d(self, scope, X, F, K, S=2, padding='SAME', phase_train=None, summary=True, BN=True):
        """
        This is a wrapper for 3-dimensional convolutions
        :param scope:
        :param X: Output of the prior layer
        :param F: Convolutional filter size
        :param K: Number of feature maps
        :param S: Stride
        :param padding: 'SAME' or 'VALID'
        :param phase_train: For batch norm implementation
        :param summary: whether to produce a tensorboard summary of this layer
        :param BN: whether to perform batch normalization
        :return:
        """

        # Set channel size based on input depth
        C = X.get_shape().as_list()[-1]

        # Set the scope
        with tf.variable_scope(scope) as scope:

            # Define the Kernel. Can use Xavier init: contrib.layers.xavier_initializer())
            kernel = tf.get_variable('Weights', shape=[F, F, F, C, K],
                                     initializer=tf.contrib.layers.xavier_initializer())

            # Add to the weights collection
            tf.add_to_collection('weights', kernel)

            # Perform the actual convolution
            conv = tf.nn.conv3d(X, kernel, [1, S, S, S, 1], padding=padding)  # Create a 2D tensor with BATCH_SIZE rows

            # Apply the batch normalization. Updates weights during training phase only
            if BN:
                conv = tf.cond(phase_train,
                               lambda: tf.contrib.layers.batch_norm(conv, activation_fn=None, center=True, scale=True,
                                                            updates_collections=None, is_training=True, reuse=None,
                                                            scope=scope, decay=0.9, epsilon=1e-5),
                               lambda: tf.contrib.layers.batch_norm(conv, activation_fn=None, center=True, scale=True,
                                                            updates_collections=None, is_training=False, reuse=True,
                                                            scope=scope, decay=0.9, epsilon=1e-5))

            # Relu activation
            conv = tf.nn.relu(conv, name=scope.name)

            # Create a histogram/scalar summary of the conv1 layer
            if summary: self._activation_summary(conv)

            return conv


    def depthwise_convolution(self, scope, X, F, K, S=2, padding='SAME', phase_train=None, summary=True, BN=True):
            """
            This is a wrapper for depthwise convolutions
            :param scope:
            :param X: Output of the prior layer
            :param F: Convolutional filter size
            :param K: Number of feature maps
            :param S: Stride
            :param padding: 'SAME' or 'VALID'
            :param phase_train: For batch norm implementation
            :param summary: whether to produce a tensorboard summary of this layer
            :param BN: whether to perform batch normalization
            :return: conv: the result of everything
            """

            # Set channel size based on input depth
            C = X.get_shape().as_list()[3]

            # Set the scope
            with tf.variable_scope(scope) as scope:

                # Define the Kernel. Can use Xavier init: contrib.layers.xavier_initializer())
                kernel = tf.get_variable('Weights', shape=[F, F, C, K],
                                         initializer=tf.contrib.layers.xavier_initializer())

                # Add to the weights collection
                tf.add_to_collection('weights', kernel)

                # Perform the actual convolution
                conv = tf.nn.depthwise_conv2d(X, kernel, [1, S, S, 1], padding=padding)

                # Apply the batch normalization. Updates weights during training phase only
                if BN:
                    conv = tf.cond(phase_train,
                                   lambda: tf.contrib.layers.batch_norm(conv, activation_fn=None, center=True,
                                                                        scale=True,
                                                                        updates_collections=None, is_training=True,
                                                                        reuse=None,
                                                                        scope=scope, decay=0.9, epsilon=1e-5),
                                   lambda: tf.contrib.layers.batch_norm(conv, activation_fn=None, center=True,
                                                                        scale=True,
                                                                        updates_collections=None, is_training=False,
                                                                        reuse=True,
                                                                        scope=scope, decay=0.9, epsilon=1e-5))

                # Relu activation
                conv = tf.nn.relu(conv, name=scope.name)

                # Create a histogram/scalar summary of the conv1 layer
                if summary: self._activation_summary(conv)

                return conv


    def deconvolution(self, scope, X, F, K, S, padding='SAME', phase_train=None,
                      concat_var=None, out_shape=None, summary=True, BN=True):
        """
        This is a wrapper for De-convolutions aka fractionally strided convolutions aka transposed convolutions
        aka upconvolutions aka backwards convolutions
        :param scope: Scope of the variables created
        :param X: The input tensor, images or result of prior convolotions
        :param F: Convolution size
        :param K: Kernel ( aka channel, aka feature map) size
        :param S: Stride size
        :param padding: 'SAME' for padding, 'VALID' for no padding
        :param phase_train: Whether we are in training or testing mode
        :param concat_var: The variable aka "skip connection" to concatenate
        :param summary: whether to produce a tensorboard summary of this layer
        :param BN: whether to perform batch normalization
        :param out_shape: The shape of output. if blank just double
        :return: conv: the result of the convolution
        """

        with tf.variable_scope(scope) as scope:

            # Set channel size based on input depth
            C = X.get_shape().as_list()[3]

            # Xavier init
            kernel = tf.get_variable('Weights', shape=[F, F, K, C], initializer=tf.contrib.layers.xavier_initializer())

            # Define the output shape if not given
            if out_shape is None:
                out_shape = X.get_shape().as_list()
                out_shape[1] *= 2
                out_shape[2] *= 2
                out_shape[3] = K

            # Perform the deconvolution. output_shape: A 1-D Tensor representing the output shape of the deconvolution op.
            dconv = tf.nn.conv2d_transpose(X, kernel, output_shape=out_shape, strides=[1, S, S, 1], padding=padding)

            # Concatenate along the depth axis
            conv = tf.concat([concat_var, dconv], axis=3)

            # Apply the batch normalization. Updates weights during training phase only
            if BN:
                conv = tf.cond(phase_train,
                               lambda: tf.contrib.layers.batch_norm(conv, activation_fn=None, center=True, scale=True,
                                                                    updates_collections=None, is_training=True,
                                                                    reuse=None,
                                                                    scope=scope, decay=0.9, epsilon=1e-5),
                               lambda: tf.contrib.layers.batch_norm(conv, activation_fn=None, center=True, scale=True,
                                                                    updates_collections=None, is_training=False,
                                                                    reuse=True,
                                                                    scope=scope, decay=0.9, epsilon=1e-5))

            # Relu
            conv = tf.nn.relu(conv, name=scope.name)

            # Create a histogram summary and summary of sparsity
            if summary: self._activation_summary(conv)

            return conv


    def inception_layer(self, scope, X, K, S=1, padding='SAME', phase_train=None, summary=True, BN=True):
        """
        This function implements an inception layer or "network within a network"
        :param scope:
        :param X: Output of the previous layer
        :param K: Feature maps in the inception layer (will be multiplied by 4 during concatenation)
        :param S: Stride
        :param padding:
        :param phase_train: For batch norm implementation
        :param summary: whether to produce a tensorboard summary of this layer
        :param BN: whether to perform batch normalization
        :return: the inception layer output after concat
        """

        # Implement an inception layer here ----------------
        with tf.variable_scope(scope) as scope:
            # First branch, 1x1x64 convolution
            inception1 = self.convolution('Inception1', X, 1, K, S,
                                          phase_train=phase_train, summary=summary, BN=BN)  # 64x64x64

            # Second branch, 1x1 convolution then 3x3 convolution
            inception2a = self.convolution('Inception2a', X, 1, 1, 1,
                                           phase_train=phase_train, summary=summary, BN=BN)  # 64x64x1

            inception2 = self.convolution('Inception2', inception2a, 3, K, S,
                                          phase_train=phase_train, summary=summary, BN=BN)  # 64x64x64

            # Third branch, 1x1 convolution then 5x5 convolution:
            inception3a = self.convolution('Inception3a', X, 1, 1, 1,
                                           phase_train=phase_train, summary=summary, BN=BN)  # 64x64x1

            inception3 = self.convolution('Inception3', inception3a, 5, K, S,
                                          phase_train=phase_train, summary=summary, BN=BN)  # 64x64x64

            # Fourth branch, max pool then 1x1 conv:
            inception4a = tf.nn.max_pool(X, [1, 3, 3, 1], [1, 1, 1, 1], padding)  # 64x64x256

            inception4 = self.convolution('Inception4', inception4a, 1, K, S,
                                          phase_train=phase_train, summary=summary, BN=BN)  # 64x64x64

            # Concatenate the results for dimension of 64,64,256
            inception = tf.concat([tf.concat([tf.concat([inception1, inception2], axis=3),
                                              inception3], axis=3), inception4], axis=3)

            return inception


    def incepted_downsample(self, scope, X, K, S=2, padding='SAME', phase_train=None, summary=True, BN=True):
        """
        This function implements a downsampling layer that gives the network 3 options
         for reducing parameter size, maxPool, avgPool or strided convolutions
        :param scope:
        :param X: Output of the previous layer
        :param K: Feature maps in the inception layer (will be multiplied by 4 during concatenation)
        :param S: The degree of downsampling or stride
        :param padding:
        :param phase_train: For batch norm implementation
        :param summary: whether to produce a tensorboard summary of this layer
        :param BN: whether to perform batch normalization
        :return: the inception layer output after concat
        """

        # Implement an inception layer here ----------------
        with tf.variable_scope(scope) as scope:

            # First branch strided convolution
            inception1 = self.convolution('Inception1', X, 3, K, S,
                                          phase_train=phase_train, summary=False, BN=False, relu=False)

            # Second branch, AVG pool
            inception2 = tf.nn.avg_pool(X, [1, 3, 3, 1], [1, S, S, 1], padding)


            # Third branch, max pool
            inception3 = tf.nn.max_pool(X, [1, 3, 3, 1], [1, S, S, 1], padding)


            # Concatenate the results
            inception = tf.concat([tf.concat([inception1, inception2], axis=3),inception3], axis=3)

            # Apply the batch normalization. Updates weights during training phase only
            if BN:
                inception = tf.cond(phase_train,
                                   lambda: tf.contrib.layers.batch_norm(inception, activation_fn=None, center=True,
                                                                        scale=True,
                                                                        updates_collections=None, is_training=True,
                                                                        reuse=None,
                                                                        scope=scope, decay=0.9, epsilon=1e-5),
                                   lambda: tf.contrib.layers.batch_norm(inception, activation_fn=None, center=True,
                                                                        scale=True,
                                                                        updates_collections=None, is_training=False,
                                                                        reuse=True,
                                                                        scope=scope, decay=0.9, epsilon=1e-5))

            # Relu activation
            inception = tf.nn.relu(inception, name=scope.name)

            # Create a histogram/scalar summary of the conv1 layer
            if summary: self._activation_summary(inception)

            return inception


    def residual_layer(self, scope, X, F, K, padding='SAME', phase_train=None, summary=True, BN=True):
        """
        This is a wrapper for implementing a hybrid residual layer with inception layer as F(x)
        :param scope:
        :param X: Output of the previous layer
        :param F: Dimensions of the second convolution in F(x) - the non inception layer one
        :param K: Feature maps in the inception layer (will be multiplied by 4 during concatenation)
        :param S: Stride
        :param padding:
        :param phase_train: For batch norm implementation
        :param summary: whether to produce a tensorboard summary of this layer
        :param BN: whether to perform batch normalization
        :return:
        """

        # Set channel size based on input depth
        C = X.get_shape().as_list()[3]

        # Set the scope. Implement a residual layer below: Conv-relu-conv-residual-relu
        with tf.variable_scope(scope) as scope:

            # The first layer is an inception layer
            conv1 = self.inception_layer(scope, X, K, 1, phase_train=phase_train)

            # Define the Kernel for conv2. Which is a normal conv layer
            kernel = tf.get_variable('Weights', shape=[F, F, C, K * 4],
                                     initializer=tf.contrib.layers.xavier_initializer())

            # Add this kernel to the weights collection for L2 reg
            tf.add_to_collection('weights', kernel)

            # Perform the actual convolution
            conv2 = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1],
                                 padding=padding)  # Create a 2D tensor with BATCH_SIZE rows

            # Add in the residual here
            residual = tf.add(conv2, X)

            # Apply the batch normalization. Updates weights during training phase only
            if BN:
                residual = tf.cond(phase_train,
                               lambda: tf.contrib.layers.batch_norm(residual, activation_fn=None, center=True, scale=True,
                                                            updates_collections=None, is_training=True, reuse=None,
                                                            scope=scope, decay=0.9, epsilon=1e-5),
                               lambda: tf.contrib.layers.batch_norm(residual, activation_fn=None, center=True, scale=True,
                                                            updates_collections=None, is_training=False, reuse=True,
                                                            scope=scope, decay=0.9, epsilon=1e-5))

            # Relu activation
            conv = tf.nn.relu(residual, name=scope.name)

            # Create a histogram/scalar summary of the conv1 layer
            if summary: self._activation_summary(conv)

            return conv


    def _activation_summary(self, x):
        """ 
        Helper to create summaries for activations
            Creates a summary to measure the proportion of your W in x that is all zero
            Parameters: x = a tensor
            Returns: Nothing
        """

        # Output a summary protobuf with a histogram of x
        tf.summary.histogram(x.op.name + '/activations', x)

        # " but with a scalar of the fraction of 0's
        tf.summary.scalar(x.op.name + '/sparsity', tf.nn.zero_fraction(x))

        return


    def fc7_layer(self, scope, X, neurons, dropout=False, phase_train=True, keep_prob=0.5, summary=True):
        """
        Wrapper for implementing a fully connected layer
        :param scope: Scopename of the layer
        :param X: Input of the prior layer
        :param neurons: Desired number of neurons in the layer
        :param dropout: Whether to implement dropout here
        :param phase_train: Are we in testing or training phase = only relevant for dropout
        :param keep_prob: if doing dropout, the keep probability
        :param summary: Whether to output a summary
        :return: fc7: the result of all of the above
        """

        # The Fc7 layer scope
        with tf.variable_scope(scope) as scope:

            # Retreive the batch size of the last layer
            batch_size = X.get_shape().as_list()[0]

            # Flatten the input layer
            reshape = tf.reshape(X, [batch_size, -1])

            # Retreive the number of columns in the flattened layer
            dim = reshape.get_shape()[1].value

            # Initialize the weights
            weights = tf.get_variable('weights', shape=[dim, neurons],
                                      initializer=tf.truncated_normal_initializer(stddev=5e-2))

            # Add to the collection of weights
            tf.add_to_collection('weights', weights)

            # Initialize the biases
            biases = tf.Variable(np.ones(neurons), name='Bias', dtype=tf.float32)

            # Do the math
            fc7 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

            # Dropout here if wanted and in train phase
            if phase_train and dropout: fc7 = tf.nn.dropout(fc7, keep_prob)

            # Activation summary
            if summary: self._activation_summary(fc7)

            return fc7


    def linear_layer(self, scope, X, neurons, dropout=False, phase_train=True, keep_prob=0.5, summary=True):
        """
        Wrapper for implementing a fully connected layer
        :param scope: Scopename of the layer
        :param X: Input of the prior layer
        :param neurons: Desired number of neurons in the layer
        :param dropout: Whether to implement dropout here
        :param phase_train: Are we in testing or training phase
        :param keep_prob: if doing dropout, the keep probability
        :param summary: Whether to output a summary
        :return: fc7: the result of all of the above
        """

        # Retreive the size of the last layer
        dim = X.get_shape().as_list()[-1]

        # The linear layer Dimensions:
        with tf.variable_scope(scope) as scope:

            # Initialize the weights
            weights = tf.get_variable('weights', shape=[dim, neurons],
                                      initializer=tf.truncated_normal_initializer(stddev=5e-2))

            # Add to the collection of weights
            tf.add_to_collection('weights', weights)

            # Initialize the biases
            biases = tf.Variable(np.ones(neurons), name='Bias', dtype=tf.float32)

            # Do the math
            linear = tf.nn.relu(tf.matmul(X, weights) + biases, name=scope.name)

            # Dropout here if wanted and in train phase
            if phase_train and dropout: linear = tf.nn.dropout(linear, keep_prob)

            # Activation summary
            if summary: self._activation_summary(linear)

            return linear


    """
         Loss function wrappers
    """

    def segmentation_SCE_loss(self, logits, labelz, class_factor=1, summary=True):
        """
        Calculates cross entropy for a segmentation type network. Made for Unet segmentation of lung nodules
        :param logits: logits from the forward pass. (batch, H, W, 1)
        :param labelz: The true input labels    (batch x H x W x 1)
        :param class_factor: For class sensitive loss functions
        :return: loss: The calculated softmax cross entropy
        """

        # First create a class sensitive nodule mask of all values > 1 (aka all the nodules)
        nodule_mask = tf.cast(labelz > 1, tf.float32)

        # Now multiply this mask by our scaling factor (hyperparameter) and add to the original mask.
        # After this point all nodules = 2+factor, lung = 1, background = 0
        nodule_mask = tf.add(tf.multiply(nodule_mask, class_factor), labelz)

        # Change the labels to one hot. Result: N x H x W x C x 1
        labels = tf.one_hot(tf.cast(labelz, tf.uint8), depth=2, dtype=tf.uint8)

        # Remove dimensions of size 1. Result: N x H x W x C
        labels = tf.squeeze(labels)

        # Calculate the loss: Result is batch x 65k
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        # Add the nodule mask factor here
        loss = tf.multiply(loss, tf.squeeze(nodule_mask))

        # Apply some Peter magic to nullify the background label
        loss = tf.multiply(loss, tf.cast(tf.squeeze(labelz) > 0, tf.float32))

        # Reduce the loss into a scalar
        loss = tf.reduce_mean(loss)

        # Output the summary
        if summary: tf.summary.scalar('loss', loss)

        # Add these losses to the collection
        tf.add_to_collection('losses', loss)

        return loss


    def MSE_loss(self, logits, labels, summary=True):
        """
        Calculates the mean squared error, made for boneAge linear regressor output. 
        :param logits: not really logits but outputs of the network
        :param labels: actual values
        :return: loss: The loss value as a tf.float32
        """

        # Calculate MSE loss: square root of the mean of the square of an elementwise subtraction of logits and labels
        MSE_loss = tf.reduce_mean(tf.square(labels - logits))

        # Output the summary of the MSE and MAE
        if summary: tf.summary.scalar('Mean Square Error', MSE_loss)

        # Add these losses to the collection
        tf.add_to_collection('losses', MSE_loss)

        # For now return MSE loss, add L2 regularization below later
        return MSE_loss


    def SCE_loss(self, logits, labels, num_classes, summary=True):
        """
        Calculates the softmax cross entropy loss between logits and labels
        :param logits:
        :param labels:
        :param num_classes: the number of classes
        :param summary: whether to output a summary of the loss to tensorboard
        :return:
        """
        # Change labels to one hot
        labels = tf.one_hot(tf.cast(labels, tf.uint8), depth=num_classes, dtype=tf.uint8)

        # Calculate  loss
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        # Reduce to scalar
        loss = tf.reduce_mean(loss)

        # Output the summary of the MSE and MAE
        if summary: tf.summary.scalar('Cross Entropy', loss)

        # Add these losses to the collection
        tf.add_to_collection('losses', loss)

        return loss
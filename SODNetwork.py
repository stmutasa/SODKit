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
        C = X.get_shape().as_list()[-1]

        # Set the scope
        with tf.variable_scope(scope) as scope:

            # Define the Kernel. Can use Xavier init: contrib.layers.xavier_initializer())
            kernel = tf.get_variable('Weights', shape=[F, F, C, K],
                                     initializer=tf.truncated_normal_initializer)

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


    def convolution_3d(self, scope, X, F, K, S=2, padding='SAME', phase_train=None, summary=True, BN=True, relu=True):
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
            if relu: conv = tf.nn.relu(conv, name=scope.name)

            # Create a histogram/scalar summary of the conv1 layer
            if summary: self._activation_summary(conv)

            return conv


    def depthwise_convolution(self, scope, X, F, K, S=2, padding='SAME', phase_train=None, summary=True, BN=True, relu=True):
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
                if relu: conv = tf.nn.relu(conv, name=scope.name)

                # Create a histogram/scalar summary of the conv1 layer
                if summary: self._activation_summary(conv)

                return conv


    def deconvolution(self, scope, X, F, K, S, padding='SAME', phase_train=None,
                      concat_var=None, out_shape=None, summary=True, BN=True, relu=True):
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
            if relu: conv = tf.nn.relu(conv, name=scope.name)

            # Create a histogram summary and summary of sparsity
            if summary: self._activation_summary(conv)

            return conv


    def inception_layer(self, scope, X, K, S=1, padding='SAME', phase_train=None, summary=True, BN=True, relu=True):
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
                                          phase_train=phase_train, summary=summary, BN=BN, relu=relu)  # 64x64x64

            # Second branch, 1x1 convolution then 3x3 convolution
            inception2a = self.convolution('Inception2a', X, 1, 1, 1,
                                           phase_train=phase_train, summary=summary)  # 64x64x1

            inception2 = self.convolution('Inception2', inception2a, 3, K, S,
                                          phase_train=phase_train, summary=summary, BN=BN, relu=relu)  # 64x64x64

            # Third branch, 1x1 convolution then 5x5 convolution:
            inception3a = self.convolution('Inception3a', X, 1, 1, 1,
                                           phase_train=phase_train, summary=summary)  # 64x64x1

            inception3 = self.convolution('Inception3', inception3a, 5, K, S,
                                          phase_train=phase_train, summary=summary, BN=BN, relu=relu)  # 64x64x64

            # Fourth branch, max pool then 1x1 conv:
            inception4a = tf.nn.max_pool(X, [1, 3, 3, 1], [1, 1, 1, 1], padding)  # 64x64x256

            inception4 = self.convolution('Inception4', inception4a, 1, K, S,
                                          phase_train=phase_train, summary=summary, BN=BN, relu=relu)  # 64x64x64

            # Concatenate the results for dimension of 64,64,256
            inception = tf.concat([inception1, inception2, inception3, inception4], axis=-1)

            return inception


    def incepted_downsample(self, scope, X, K, S=2, padding='SAME', phase_train=None, summary=True, BN=True, relu=True):
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

            # x: (-, 8, 8, k), strided: (-, 4, 4, k), avg: (-, 4, 4, k), max: (-, 4, 4, k), inc: (11, 4, 4, k*4)

            # First branch strided convolution
            strided = self.convolution('IDS1', X, 2, K, S,
                                          phase_train=phase_train, summary=False, BN=False, relu=False)

            # 1x1 convolution to match filters
            X = self.convolution('1by1', X, 1, K, 1, 'SAME', phase_train, summary, False, False)

            # Second branch, AVG pool
            avg = tf.nn.avg_pool(X, [1, 2, 2, 1], [1, S, S, 1], padding)

            # Third branch, max pool
            max = tf.nn.max_pool(X, [1, 2, 2, 1], [1, S, S, 1], padding)

            # Concatenate the results
            inception = tf.concat([tf.concat([strided, avg], axis=-1),max], axis=-1)

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
            if relu: inception = tf.nn.relu(inception, name=scope.name)

            # Create a histogram/scalar summary of the conv1 layer
            if summary: self._activation_summary(inception)

            return inception


    def res_inc_layer(self, scope, X, F, K, padding='SAME', phase_train=None, summary=True, BN=True, relu=True):
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
            conv1 = self.inception_layer(scope, X, K/8, 1, phase_train=phase_train)

            # Set channel size based on input depth
            #C = conv1.get_shape().as_list()[3]

            # Define the Kernel for conv2. Which is a normal conv layer
            kernel = tf.get_variable('Weights', shape=[F, F, C, K],
                                     initializer=tf.contrib.layers.xavier_initializer())

            # Add this kernel to the weights collection for L2 reg
            tf.add_to_collection('weights', kernel)

            # Perform the actual convolution
            conv2 = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding=padding)

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
            if relu: residual = tf.nn.relu(residual, name=scope.name)

            # Create a histogram/scalar summary of the conv1 layer
            if summary: self._activation_summary(residual)

            return residual


    def residual_layer(self, scope, X, F, K, S=2, K_prob=None, padding='SAME',
                       phase_train=None, summary=True, DSC=False, BN=False, relu=False):
        """
        This is a wrapper for implementing a stanford style residual layer
        :param scope:
        :param X: Output of the previous layer, make sure this is not normalized and has no nonlinearity applied
        :param F: Dimensions of the second convolution in F(x) - the non inception layer one
        :param K: Feature maps in the inception layer (will be multiplied by 4 during concatenation)
        :param S: Stride of the convolution, whether to downsample
        :param k_prob: keep probability for dropout
        :param padding: SAME or VALID
        :param phase_train: For batch norm implementation
        :param summary: whether to produce a tensorboard summary of this layer
        :param DSC: Whether to perform standard downsample or incepted downsample
        :param BN: Whether to batch norm. Defaults to false to plug into another residual
        :param relu: whether to apply a nonlinearity at the end.
        :return:
        """

        # Set the scope. Implement a residual layer below
        with tf.variable_scope(scope) as scope:

            # Start the second conv layer: BN->Relu->Drop->Conv->BN->Relu->dropout
            conv1 = self.batch_normalization(X, phase_train, scope)

            # ReLU
            conv1 = tf.nn.relu(conv1, scope.name)

            # Dropout
            if K_prob: conv1 = tf.nn.dropout(conv1, K_prob)

            # Another Convolution with BN and ReLu
            conv2 = self.convolution('Conv2', conv1, F, K, 1, 'SAME', phase_train, summary, True, True)

            # Second dropout
            if K_prob: conv2 = tf.nn.dropout(conv2, K_prob)

            # Final STRIDED conv without BN or RELU
            if DSC: conv = self.incepted_downsample('ConvFinal', conv2, K, S, 'SAME', phase_train, summary, False, False)
            else: conv = self.convolution('ConvFinal', conv2, F, K, S, 'SAME', phase_train, summary, False, False)

            # 1x1 convolution to match filters
            if DSC: X = self.convolution('1by1', X, 1, K*3, 1, 'SAME', phase_train, summary, False, False)
            else: X = self.convolution('1by1', X, 1, K, 1, 'SAME', phase_train, summary, False, False)

            # Max pool of the input convolution
            pool = tf.nn.max_pool(X, [1, F, F, 1], [1, S, S, 1], padding)

            # The Residual block
            residual = tf.add(conv, pool)

            # Apply the batch normalization. Updates weights during training phase only
            if BN:
                residual = tf.cond(phase_train,
                                   lambda: tf.contrib.layers.batch_norm(residual, activation_fn=None, center=True,
                                                                        scale=True,
                                                                        updates_collections=None, is_training=True,
                                                                        reuse=None,
                                                                        scope='BNRes', decay=0.9, epsilon=1e-5),
                                   lambda: tf.contrib.layers.batch_norm(residual, activation_fn=None, center=True,
                                                                        scale=True,
                                                                        updates_collections=None, is_training=False,
                                                                        reuse=True,
                                                                        scope='BNRes', decay=0.9, epsilon=1e-5))

            # Relu activation
            if relu: residual = tf.nn.relu(residual, name=scope.name)

            return residual


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


    def fc7_layer(self, scope, X, neurons, dropout=False, phase_train=True, keep_prob=0.5, summary=True, BN=False):
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
            fc7 = tf.matmul(reshape, weights)
            if BN: fc7 = self.batch_normalization(fc7, phase_train, 'Fc7Norm')
            fc7 = tf.nn.relu(fc7 + biases, name=scope.name)

            # Dropout here if wanted and in train phase
            if phase_train and dropout: fc7 = tf.nn.dropout(fc7, keep_prob)

            # Activation summary
            if summary: self._activation_summary(fc7)

            return fc7


    def linear_layer(self, scope, X, neurons, dropout=False, phase_train=True, keep_prob=0.5, summary=True, BN=False):
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
                                      initializer=tf.contrib.layers.xavier_initializer())

            # Add to the collection of weights
            tf.add_to_collection('weights', weights)

            # Initialize the biases
            biases = tf.Variable(np.zeros(neurons), name='Bias', dtype=tf.float32)

            # Do the math
            # Do the math
            linear = tf.matmul(X, weights)
            if BN: linear = self.batch_normalization(linear, phase_train, 'LinearNorm')
            linear = tf.nn.relu(linear + biases, name=scope.name)

            # Dropout here if wanted and in train phase
            if phase_train and dropout: linear = tf.nn.dropout(linear, keep_prob)

            # Activation summary
            if summary: self._activation_summary(linear)

            return linear


    def spatial_transform_layer(self, scope, X):
        """
        Spatial transformer network implementation
        :param scope: Scope
        :param X: The input tensor
        :return: h_trans: the output of the transformer
        """

        dim = X.get_shape().as_list()[1]
        batch_size = X.get_shape().as_list()[0]

        with tf.variable_scope(scope) as scope:

            # Set up the localisation network to calculate floc(u):
            W1 = tf.get_variable('Weights1', shape=[dim * dim * 128, 20],
                                 initializer=tf.truncated_normal_initializer(stddev=5e-2))
            B1 = tf.get_variable('Bias1', shape=[20], initializer=tf.truncated_normal_initializer(stddev=5e-2))
            W2 = tf.get_variable('Weights2', shape=[20, 6], initializer=tf.truncated_normal_initializer(stddev=5e-2))

            # Add weights to collection
            tf.add_to_collection('weights', W1)
            tf.add_to_collection('weights', W2)

            # Always start with the identity transformation
            initial = np.array([[1.0, 0, 0], [0, 1.0, 0]])
            initial = initial.astype('float32')
            initial = initial.flatten()
            B2 = tf.Variable(initial_value=initial, name='Bias2')

            # Define the two layers of the localisation network
            H1 = tf.nn.tanh(tf.matmul(tf.zeros([batch_size, dim * dim * 128]), W1) + B1)
            H2 = tf.nn.tanh(tf.matmul(H1, W2) + B2)

            # Define the output size to the original dimensions
            output_size = (dim, dim)
            h_trans = self.transformer(X, H2, output_size)

            return h_trans


    def transition_layer(self, scope, X, K, S=1, padding='SAME', phase_train=None, summary=True, BN=True, relu=True):
        """
        This function implements a transition layer to insert before the FC layer. Improves regularization
        :param scope:
        :param X: Output of the previous layer
        :param K: Feature maps in the inception layer (will be multiplied by 4 during concatenation)
        :param S: Stride Whether to downsample the convoluted inceptions
        :param padding:
        :param phase_train: For batch norm implementation
        :param summary: whether to produce a tensorboard summary of this layer
        :param BN: whether to perform batch normalization
        :return: the inception layer output after concat
        """

        # Implement an incepted transition layer here ----------------
        with tf.variable_scope(scope) as scope:

            # Retreive size of prior network. Prior = [batch, F, F, K]
            F = X.get_shape().as_list()[1]

            # First branch, 7x7 conv then global avg pool
            inception1a = self.convolution('Transition1', X, 7, K, S, 'SAME', phase_train, summary, BN, relu)
            inception1 = tf.nn.avg_pool(inception1a, [1, F, F, 1], [1, 2, 2, 1], 'VALID')

            # Second branch, 5x5 conv then global avg pool
            inception2a = self.convolution('Transition2', X, 5, K, S, 'SAME', phase_train, summary, BN, relu)
            inception2 = tf.nn.avg_pool(inception2a, [1, F, F, 1], [1, 2, 2, 1], 'VALID')

            # Third branch, 3x3 conv then global avg pool
            inception3a = self.convolution('Transition3', X, 3, K, S, 'SAME', phase_train, summary, BN, relu)
            inception3 = tf.nn.avg_pool(inception3a, [1, F, F, 1], [1, 2, 2, 1], 'VALID')

            # Concatenate the results
            inception = tf.concat([inception1, inception2, inception3], axis=-1)

            return tf.squeeze(inception)


    """
         Loss function wrappers
    """

    def segmentation_SCE_loss(self, logits, labelz, class_factor=1.2, summary=True):
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


    def cost_sensitive_loss(self, logits, labels, loss_factor, num_classes):
        """
        For calculating a class sensitive loss
        :param logits: the predictions of the network
        :param labels: the ground truth
        :param loss_factor: "extra" penalty for missing this class
        :param num_classes: number of classes
        :return:
        """

        # Make a nodule sensitive binary for values > 0 (aka all the actual cancers)
        lesion_mask = tf.cast(labels > 0, tf.float32)

        # Now multiply this mask by scaling factor then add back to labels. Add 1 to prevent 0 loss
        lesion_mask = tf.add(tf.multiply(lesion_mask, loss_factor), 1)

        # Change labels to one hot
        labels = tf.one_hot(tf.cast(labels, tf.uint8), depth=num_classes, dtype=tf.uint8)

        # Calculate  loss
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.squeeze(labels), logits=logits)

        # Multiply the loss factor
        loss = tf.multiply(loss, tf.squeeze(lesion_mask))

        # Reduce to scalar
        loss = tf.reduce_mean(loss)

        # Output the summary of the MSE and MAE
        tf.summary.scalar('Cross Entropy', loss)

        # Add these losses to the collection
        tf.add_to_collection('losses', loss)

        return loss


    """
            Utility Functions
    """

    def batch_normalization(self, conv, phase_train, scope):
        """
        Wrapper for the batch normalization implementation
        :param conv: The layer to normalize
        :param phase_train: Is this training or testing phase
        :param scope: the scope for this operation
        :return: conv: the normalized convolution
        """

        return tf.cond(phase_train,
                       lambda: tf.contrib.layers.batch_norm(conv, activation_fn=None, center=True, scale=True,
                                                    updates_collections=None, is_training=True, reuse=None,
                                                    scope=scope, decay=0.9, epsilon=1e-5),
                       lambda: tf.contrib.layers.batch_norm(conv, activation_fn=None, center=True, scale=True,
                                                    updates_collections=None, is_training=False, reuse=True,
                                                    scope=scope, decay=0.9, epsilon=1e-5))


    def transformer(self, U, theta, out_size, name='SpatialTransformer', **kwargs):
        """Spatial Transformer Layer

        Implements a spatial transformer layer as described in [1]_.
        Based on [2]_ and edited by David Dao for Tensorflow.

        Parameters
        ----------
        U : float
            The output of a convolutional net should have the
            shape [num_batch, height, width, num_channels].
        theta: float
            The output of the
            localisation network should be [num_batch, 6].
        out_size: tuple of two ints
            The size of the output of the network (height, width)

        References
        ----------
        .. [1]  Spatial Transformer Networks
                Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
                Submitted on 5 Jun 2015
        .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

        Notes
        -----
        To initialize the network to the identity transform init
        ``theta`` to :
            identity = np.array([[1., 0., 0.],
                                 [0., 1., 0.]])
            identity = identity.flatten()
            theta = tf.Variable(initial_value=identity)

        """

        def _repeat(x, n_repeats):
            with tf.variable_scope('_repeat'):
                rep = tf.transpose(
                    tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
                rep = tf.cast(rep, 'int32')
                x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
                return tf.reshape(x, [-1])

        def _interpolate(im, x, y, out_size):
            with tf.variable_scope('_interpolate'):
                # constants
                num_batch = im.get_shape().as_list()[0]
                height = im.get_shape().as_list()[1]
                width = im.get_shape().as_list()[2]
                channels = im.get_shape().as_list()[3]

                x = tf.cast(x, 'float32')
                y = tf.cast(y, 'float32')
                height_f = tf.cast(height, 'float32')
                width_f = tf.cast(width, 'float32')
                out_height = out_size[0]
                out_width = out_size[1]
                zero = tf.zeros([], dtype='int32')
                max_y = tf.cast(im.get_shape().as_list()[1] - 1, 'int32')
                max_x = tf.cast(im.get_shape().as_list()[2] - 1, 'int32')

                # scale indices from [-1, 1] to [0, width/height]
                x = (x + 1.0) * (width_f) / 2.0
                y = (y + 1.0) * (height_f) / 2.0

                # do sampling
                x0 = tf.cast(tf.floor(x), 'int32')
                x1 = x0 + 1
                y0 = tf.cast(tf.floor(y), 'int32')
                y1 = y0 + 1

                x0 = tf.clip_by_value(x0, zero, max_x)
                x1 = tf.clip_by_value(x1, zero, max_x)
                y0 = tf.clip_by_value(y0, zero, max_y)
                y1 = tf.clip_by_value(y1, zero, max_y)
                dim2 = width
                dim1 = width * height
                base = _repeat(tf.range(num_batch) * dim1, out_height * out_width)
                base_y0 = base + y0 * dim2
                base_y1 = base + y1 * dim2
                idx_a = base_y0 + x0
                idx_b = base_y1 + x0
                idx_c = base_y0 + x1
                idx_d = base_y1 + x1

                # use indices to lookup pixels in the flat image and restore
                # channels dim
                im_flat = tf.reshape(im, tf.stack([-1, channels]))
                im_flat = tf.cast(im_flat, 'float32')
                Ia = tf.gather(im_flat, idx_a)
                Ib = tf.gather(im_flat, idx_b)
                Ic = tf.gather(im_flat, idx_c)
                Id = tf.gather(im_flat, idx_d)

                # and finally calculate interpolated values
                x0_f = tf.cast(x0, 'float32')
                x1_f = tf.cast(x1, 'float32')
                y0_f = tf.cast(y0, 'float32')
                y1_f = tf.cast(y1, 'float32')
                wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
                wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
                wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
                wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
                output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
                return output

        def _meshgrid(height, width):
            with tf.variable_scope('_meshgrid'):
                # This should be equivalent to:
                #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
                #                         np.linspace(-1, 1, height))
                #  ones = np.ones(np.prod(x_t.shape))
                #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
                x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                                tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
                y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                                tf.ones(shape=tf.stack([1, width])))

                x_t_flat = tf.reshape(x_t, (1, -1))
                y_t_flat = tf.reshape(y_t, (1, -1))

                ones = tf.ones_like(x_t_flat)
                grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
                return grid

        def _transform(theta, input_dim, out_size):
            with tf.variable_scope('_transform'):
                num_batch = input_dim.get_shape().as_list()[0]
                height = input_dim.get_shape().as_list()[1]
                width = input_dim.get_shape().as_list()[2]
                num_channels = input_dim.get_shape().as_list()[3]
                theta = tf.reshape(theta, (-1, 2, 3))
                theta = tf.cast(theta, 'float32')

                # grid of (x_t, y_t, 1), eq (1) in ref [1]
                height_f = tf.cast(height, 'float32')
                width_f = tf.cast(width, 'float32')
                out_height = out_size[0]
                out_width = out_size[1]
                grid = _meshgrid(out_height, out_width)
                grid = tf.expand_dims(grid, 0)
                grid = tf.reshape(grid, [-1])
                grid = tf.tile(grid, tf.stack([num_batch]))
                grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))

                # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
                T_g = tf.matmul(theta, grid)
                x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
                y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
                x_s_flat = tf.reshape(x_s, [-1])
                y_s_flat = tf.reshape(y_s, [-1])

                input_transformed = _interpolate(
                    input_dim, x_s_flat, y_s_flat,
                    out_size)

                output = tf.reshape(
                    input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
                return output

        with tf.variable_scope(name):
            output = _transform(theta, U, out_size)
            return output


    def batch_transformer(self, U, thetas, out_size, name='BatchSpatialTransformer'):
        """Batch Spatial Transformer Layer

        Parameters
        ----------

        U : float
            tensor of inputs [num_batch,height,width,num_channels]
        thetas : float
            a set of transformations for each input [num_batch,num_transforms,6]
        out_size : int
            the size of the output [out_height,out_width]

        Returns: float
            Tensor of size [num_batch*num_transforms,out_height,out_width,num_channels]
        """
        with tf.variable_scope(name):
            num_batch, num_transforms = map(int, thetas.get_shape().as_list()[:2])
            indices = [[i] * num_transforms for i in range(num_batch)]
            input_repeated = tf.gather(U, tf.reshape(indices, [-1]))
            return self.transformer(input_repeated, thetas, out_size)

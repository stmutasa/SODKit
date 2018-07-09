"""
SOD Network contains function wrappers for various tensorflow tasks including:

convolutions, deconvolutions, 3D convolutions, 

"""

import tensorflow as tf
import numpy as np


class SODMatrix(object):

    #  SOD Loader class is a class for loading all types of data into protocol buffers


    # Define training or testing phase
    training_phase = None

    def __init__(self, summary=True, phase_train=True):

        self.summary=summary
        self.phase_train=phase_train

        pass


    # *************** Convolution wrappers ***************


    def convolution(self, scope, X, F, K, S=2, padding='SAME', phase_train=None, BN=True, relu=True, downsample=False, bias=True, dropout=None):
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
        :param downsample: whether to perform a max/avg downsampling at the end
        :param bias: whether to include a bias term
        :param dropout: whether to use dropout
        :return:
        """

        # Set channel size based on input depth
        C = X.get_shape().as_list()[-1]
        B = X.get_shape().as_list()[0]

        # Set the scope
        with tf.variable_scope(scope) as scope:

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

            # Channel wise dropout
            if dropout and phase_train==True: conv = tf.nn.dropout(conv, dropout, noise_shape=[B, 1, 1, C])

            # If requested, use the avg + max pool downsample operation
            if downsample: conv = self.incepted_downsample(conv)

            # Create a histogram/scalar summary of the conv1 layer
            if self.summary: self._activation_summary(conv)

            return conv


    def convolution_3d(self, scope, X, F, K, S=2, padding='SAME', phase_train=None, BN=True, relu=True, downsample=False, dropout=None):
        """
        This is a wrapper for 3-dimensional convolutions
        :param scope:
        :param X: Output of the prior layer
        :param F: Convolutional filter size
        :param K: Number of feature maps
        :param S: Stride
        :param padding: 'SAME' or 'VALID'
        :param phase_train: For batch norm implementation
        :param BN: whether to perform batch normalization
        :param relu: Whether to perform relu
        :param downsample: Whether to perform a max+avg pool downsample
        :param dropout: whether to apply channel_wise dropout
        :return:
        """

        # Set channel size based on input depth
        C = X.get_shape().as_list()[-1]
        B = X.get_shape().as_list()[0]

        # Set the scope
        with tf.variable_scope(scope) as scope:

            # Set training phase variable
            self.training_phase = phase_train

            # Define the Kernel. Can use he et al
            try: kernel = tf.get_variable('Weights', shape=[F[0], F[1], F[2], C, K],
                                     initializer=tf.contrib.layers.variance_scaling_initializer())
            except: kernel = tf.get_variable('Weights', shape=[F, F, F, C, K],
                                     initializer=tf.contrib.layers.variance_scaling_initializer())

            # Define the biases
            bias = tf.get_variable('Bias', shape=[K], initializer=tf.constant_initializer(0.0))

            # Add to the weights collection
            tf.add_to_collection('weights', kernel)
            tf.add_to_collection('biases', bias)

            # Perform the actual convolution
            try: conv = tf.nn.conv3d(X, kernel, [1, S[0], S[1], S[2], 1], padding=padding)
            except: conv = tf.nn.conv3d(X, kernel, [1, S, S, S, 1], padding=padding)

            # Add the bias
            conv = tf.nn.bias_add(conv, bias)

            # Relu activation
            if relu: conv = tf.nn.relu(conv, name=scope.name)

            # Apply the batch normalization. Updates weights during training phase only
            if BN: conv = self.batch_normalization(conv, phase_train, scope)

            # Channel wise Dropout
            if dropout and phase_train == True: conv = tf.nn.dropout(conv, dropout, noise_shape=[B, 1, 1, 1, C])

            # If requested, use the avg + max pool downsample operation
            if downsample: conv = self.incepted_downsample_3d(conv)

            # Create a histogram/scalar summary of the conv1 layer
            if self.summary: self._activation_summary(conv)

            return conv


    def depthwise_convolution(self, scope, X, F, K, S=2, padding='SAME', phase_train=None, BN=True, relu=True):
            """
            This is a wrapper for depthwise convolutions
            :param scope:
            :param X: Output of the prior layer
            :param F: Convolutional filter size
            :param K: Number of feature maps
            :param S: Stride
            :param padding: 'SAME' or 'VALID'
            :param phase_train: For batch norm implementation
            :param BN: whether to perform batch normalization
            :return: conv: the result of everything
            """

            # Set channel size based on input depth
            C = X.get_shape().as_list()[-1]

            # Set the scope
            with tf.variable_scope(scope) as scope:

                # Define the Kernel. Can use Xavier init: contrib.layers.xavier_initializer())
                kernel = tf.get_variable('Weights', shape=[F, F, C, K],
                                         initializer=tf.contrib.layers.variance_scaling_initializer())

                # Define the biases
                bias = tf.get_variable('Bias', shape=[K], initializer=tf.constant_initializer(0.0))

                # Add to the weights collection
                tf.add_to_collection('weights', kernel)
                tf.add_to_collection('biases', bias)

                # Perform the actual convolution
                conv = tf.nn.depthwise_conv2d(X, kernel, [1, S, S, 1], padding=padding)

                # Apply the batch normalization. Updates weights during training phase only
                if BN: conv = self.batch_normalization(conv, phase_train, 'DWC_Norm')

                # Add the bias
                conv = tf.nn.bias_add(conv, bias)

                # Relu activation
                if relu: conv = tf.nn.relu(conv, name=scope.name)

                # Create a histogram/scalar summary of the conv1 layer
                if self.summary: self._activation_summary(conv)

                return conv


    def deconvolution(self, scope, X, F, K, S, padding='SAME', phase_train=None, concat=True,
                      concat_var=None, out_shape=None, BN=True, relu=True):
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
        :param concat: Whether to concatenate or add
        :param concat_var: The variable aka "skip connection" to concatenate
        :param BN: whether to perform batch normalization
        :param out_shape: The shape of output. if blank just double
        :return: conv: the result of the convolution
        """

        with tf.variable_scope(scope) as scope:

            # Set channel size based on input depth
            C = X.get_shape().as_list()[-1]

            # Xavier init
            kernel = tf.get_variable('Weights', shape=[F, F, K, C],
                                     initializer=tf.contrib.layers.variance_scaling_initializer())

            # Define the biases
            bias = tf.get_variable('Bias', shape=[K], initializer=tf.constant_initializer(0.0))

            # Add to the weights collection
            tf.add_to_collection('weights', kernel)
            tf.add_to_collection('biases', bias)

            # Define the output shape if not given
            if out_shape is None:
                out_shape = X.get_shape().as_list()
                out_shape[1] *= 2
                out_shape[2] *= 2
                out_shape[3] = K

            # Perform the deconvolution. output_shape: A 1-D Tensor representing the output shape of the deconvolution op.
            conv = tf.nn.conv2d_transpose(X, kernel, output_shape=out_shape, strides=[1, S, S, 1], padding=padding)

            # Apply the batch normalization. Updates weights during training phase only
            if BN: conv = self.batch_normalization(conv, phase_train, scope)

            if concat_var is not None:

                # Concatenate or add along the depth axis
                if concat: conv = tf.concat([concat_var, conv], axis=-1)
                else: conv = tf.add(conv, concat_var)

            # Add in bias
            conv = tf.nn.bias_add(conv, bias)

            # Relu
            if relu: conv = tf.nn.relu(conv, name=scope.name)

            # Create a histogram summary and summary of sparsity
            if self.summary: self._activation_summary(conv)

            return conv


    def deconvolution_3d(self, scope, X, F, K, S, padding='SAME', phase_train=None, concat=True,
                      concat_var=None, out_shape=None, BN=True, relu=True):
        """
        This is a wrapper for 3D De-convolutions aka fractionally strided convolutions aka transposed convolutions
        aka upconvolutions aka backwards convolutions
        :param scope: Scope of the variables created
        :param X: The input tensor, images or result of prior convolotions
        :param F: Convolution size
        :param K: Kernel ( aka channel, aka feature map) size
        :param S: Stride size
        :param padding: 'SAME' for padding, 'VALID' for no padding
        :param phase_train: Whether we are in training or testing mode
        :param concat: whether to concatenate skip connection or use a residual connection
        :param concat_var: The variable aka "skip connection" to concatenate
        :param BN: whether to perform batch normalization
        :param out_shape: The shape of output. if blank just double
        :return: conv: the result of the convolution
        """

        with tf.variable_scope(scope) as scope:

            # Set channel size based on input depth
            C = X.get_shape().as_list()[-1]

            # Xavier init
            kernel = tf.get_variable('Weights', shape=[F, F, F, K, C],
                                     initializer=tf.contrib.layers.variance_scaling_initializer())

            # Define the biases
            bias = tf.get_variable('Bias', shape=[K], initializer=tf.constant_initializer(0.0))

            # Add to the weights collection
            tf.add_to_collection('weights', kernel)
            tf.add_to_collection('biases', bias)

            # Define the output shape if not given
            if out_shape is None:
                out_shape = X.get_shape().as_list()
                out_shape[1] *= 2
                out_shape[2] *= 2
                out_shape[3] *= 2
                out_shape[4] = K

            # Perform the deconvolution. output_shape: A 1-D Tensor representing the output shape of the deconvolution op.
            conv = tf.nn.conv3d_transpose(X, kernel, output_shape=out_shape, strides=[1, S, S, S, 1], padding=padding)

            # Apply the batch normalization. Updates weights during training phase only
            if BN: conv = self.batch_normalization(conv, phase_train, scope)

            # Concatenate or add along the depth axis
            if concat: conv = tf.concat([concat_var, conv], axis=-1)
            else: conv = tf.add(conv, concat_var)

            # Add in bias
            conv = tf.nn.bias_add(conv, bias)

            # Relu
            if relu: conv = tf.nn.relu(conv, name=scope.name)

            # Create a histogram summary and summary of sparsity
            if self.summary: self._activation_summary(conv)

            return conv


    def inception_layer(self, scope, X, K, S=1, padding='SAME', phase_train=None, BN=True, relu=True, dropout=None):
        """
        This function implements an inception layer or "network within a network"
        :param scope:
        :param X: Output of the previous layer
        :param K: Output maps in the inception layer
        :param S: Stride
        :param padding:
        :param phase_train: For batch norm implementation
        :param BN: whether to perform batch normalization
        :param dropout: keep probability for applying channel wise dropout
        :return: the inception layer output after concat
        """

        # Set output feature maps
        orig_K = K
        if S > 1: K = int(K/4)
        else: K = int(3*K/8)

        # Set BN and relu
        if S ==1:
            oBN = BN
            orelu=relu
            BN = True
            relu=True

        # Implement an inception layer here ----------------
        with tf.variable_scope(scope) as scope:

            # First branch, 1x1xK convolution
            inception1 = self.convolution('Inception1', X, 1, K, S, phase_train=phase_train,  BN=BN, relu=relu)

            # Second branch, 1x1 bottleneck then 3x3 convolution
            inception2 = self.convolution('Inception2a', X, 1, K, S, phase_train=phase_train)
            inception2 = self.convolution('Inception2', inception2, 3, K, 1, phase_train=phase_train, BN=BN, relu=relu)

            # Third branch, 1x1 bottleneck then two 3x3 convolutions (5x5 mimic):
            inception3 = self.convolution('Inception3a', X, 1, K, S, phase_train=phase_train)
            inception3 = self.convolution('Inception3b', inception3, 3, K, 1, phase_train=phase_train, BN=BN, relu=relu)
            inception3 = self.convolution('Inception3', inception3, 3, K, 1, phase_train=phase_train, BN=BN, relu=relu)

            # Fourth branch, max pool then 1x1 conv:
            inception4 = tf.nn.max_pool(X, [1, 3, 3, 1], [1, 1, 1, 1], padding)
            inception4 = self.convolution('Inception4', inception4, 1, K, S, phase_train=phase_train, BN=BN, relu=relu)

            # Concatenate the results
            inception = tf.concat([inception1, inception2, inception3, inception4], axis=-1)

            # Get dimensions
            C = inception.get_shape().as_list()[-1]
            B = inception.get_shape().as_list()[0]

            # Apply channel wise dropout here
            if dropout and phase_train == True: inception = tf.nn.dropout(inception, dropout, noise_shape=[B, 1, 1, C])

            # Final projection
            if S==1: inception = self.convolution('Inception_Fin', inception, 1, orig_K, 1, phase_train=phase_train,
                                                  BN=oBN, relu=orelu)


            return inception


    def inception_layer_3d(self, scope, X, K, Fz=1, S=1, padding='SAME', phase_train=None, BN=True, relu=True, dropout=None):
        """
        This function implements a 3D inception layer or "network within a network"
        :param scope:
        :param X: Output of the previous layer
        :param K: Desired output feature maps in the inception layer
        :param Fz: The z-axis conv kernel size
        :param S: Stride
        :param padding:
        :param phase_train: For batch norm implementation
        :param BN: whether to perform batch normalization
        :param dropout: keep prob for applying channel wise dropout
        :return: the inception layer output after concat
        """

        # Set output feature maps
        orig_K = K
        if S > 1: K = int(K / 4)
        else: K = int(3 * K / 8)

        # Set BN and relu
        if S == 1:
            oBN = BN
            orelu = relu
            BN = True
            relu = True

        # Implement an inception layer here ----------------
        with tf.variable_scope(scope) as scope:

            # First branch, 1x1x64 convolution
            inception1 = self.convolution_3d('Inception1', X, [Fz, 1, 1], K, S,
                                          phase_train=phase_train, BN=BN, relu=relu)

            # Second branch, 1x1 convolution then 3x3 convolution
            inception2 = self.convolution_3d('Inception2a', X, [Fz, 1, 1], K, S,
                                           phase_train=phase_train)
            inception2 = self.convolution_3d('Inception2', inception2, [1, 3, 3], K, 1,
                                          phase_train=phase_train, BN=BN, relu=relu)

            # Third branch, 1x1 convolution then 5x5 convolution:
            inception3 = self.convolution_3d('Inception3a', X, [Fz, 1, 1], K, S, phase_train=phase_train)
            inception3 = self.convolution_3d('Inception3b', inception3, [1, 3, 3], K, 1, phase_train=phase_train, BN=BN, relu=relu)
            inception3 = self.convolution_3d('Inception3', inception3, [1, 3, 3], K, 1, phase_train=phase_train, BN=BN, relu=relu)

            # Fourth branch, max pool then 1x1 conv:
            inception4 = tf.nn.max_pool3d(X, [1, Fz, 3, 3, 1], [1, 1, 1, 1, 1], padding)
            inception4 = self.convolution_3d('Inception4', inception4, 1, K, S, phase_train=phase_train, BN=BN, relu=relu)

            # Concatenate the results
            inception = tf.concat([inception1, inception2, inception3, inception4], axis=-1)

            # Get dimensions
            C = inception.get_shape().as_list()[-1]
            B = inception.get_shape().as_list()[0]

            # Apply channel wise dropout here
            if dropout and phase_train == True: inception = tf.nn.dropout(inception, dropout, noise_shape=[B, 1, 1, 1, C])

            # Final projection
            if S == 1: inception = self.convolution_3d('Inception_Fin', inception, 1, orig_K, 1,
                                                       phase_train=phase_train, BN=oBN, relu=orelu)

            return inception


    def residual_layer_stanford(self, scope, X, F, K, S=2, K_prob=None, padding='SAME', phase_train=None, DSC=False, BN=False, relu=False, dropout=None):
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
        :param DSC: Whether to perform standard downsample or incepted downsample
        :param BN: Whether to batch norm. Defaults to false to plug into another residual
        :param relu: whether to apply a nonlinearity at the end.
        :param dropout: keep prob for applying channel wise dropout
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
            conv2 = self.convolution('Conv2', conv1, F, K, 1, padding, phase_train, True, True)

            # Second dropout
            if K_prob: conv2 = tf.nn.dropout(conv2, K_prob)

            # Final downsampled conv without BN or RELU
            if DSC: conv = self.convolution('ConvFinal', conv2, F, K, 1, padding, phase_train, False, False, True)
            else: conv = self.convolution('ConvFinal', conv2, F, K*S, S, padding, phase_train, False, False)

            # Downsample the residual input if we did the conv layer. pool or strided
            if DSC: X = self.incepted_downsample(X)
            elif S>1: X = self.convolution('ResDown', X, 1, K*S, S, phase_train=phase_train, BN=False, relu=False)

            # Apply the batch normalization. Updates weights during training phase only
            if BN: conv = self.batch_normalization(conv, phase_train, scope)

            # The Residual block
            residual = tf.add(conv, X)

            # Relu activation
            if relu: residual = tf.nn.relu(residual, name=scope.name)

            # Get dimensions
            C = residual.get_shape().as_list()[-1]
            B = residual.get_shape().as_list()[0]

            # Apply channel wise dropout here
            if dropout and phase_train == True: residual = tf.nn.dropout(residual, dropout, noise_shape=[B, 1, 1, C])

            return residual


    def residual_layer(self, scope, residual, F, K, S=2, padding='SAME', phase_train=None, dropout=None):

        """
        This is a wrapper for implementing a microsoft style residual layer
        :param scope:
        :param residual: Output of the previous layer
        :param F: Dimensions of the second convolution in F(x) - the non inception layer one
        :param K: Feature maps in the inception layer (will be multiplied by 4 during concatenation)
        :param S: Stride of the convolution, whether to downsample
        :param padding: SAME or VALID
        :param phase_train: For batch norm implementation
        :param dropout: keep prob for applying channel wise dropout
        :return:
        """

        # Set the scope. Implement a residual layer below
        with tf.variable_scope(scope) as scope:

            # First Convolution with BN and ReLU
            conv = self.convolution('Conv1', residual, F, K, S, padding, phase_train, True, True)

            # Second convolution without ReLU
            conv = self.convolution('Conv2', conv, F, K, 1, padding, phase_train, True, False)

            # Downsample the residual input using strided 1x1 conv
            if S>1: residual = self.convolution('Res_down', residual, 1, K, S, padding, phase_train, True, False)

            # Add the Residual
            conv = tf.add(conv, residual)
            conv = tf.nn.relu(conv, name=scope.name)

            # Get dimensions
            C = conv.get_shape().as_list()[-1]
            B = conv.get_shape().as_list()[0]

            # Apply channel wise dropout here
            if dropout and phase_train == True: conv = tf.nn.dropout(conv, dropout, noise_shape=[B, 1, 1, C])

            return conv


    def residual_bottleneck_layer(self, scope, residual, F, K, S=2, padding='SAME', phase_train=None, dropout=None):

        """
        This is a wrapper for implementing a microsoft style residual layer with bottlenecks for deep networks
        :param scope:
        :param residual: Output of the previous layer
        :param F: Dimensions of the second convolution in F(x) - the non inception layer one
        :param K: Feature maps in the inception layer (will be multiplied by 4 during concatenation)
        :param S: Stride of the convolution, whether to downsample
        :param padding: SAME or VALID
        :param phase_train: For batch norm implementation
        :param dropout: keep prob for applying channel wise dropout
        :return:
        """

        # Set the scope. Implement a residual layer below
        with tf.variable_scope(scope) as scope:

            # First Convolution with BN and ReLU
            conv = self.convolution('Conv1', residual, 1, int(K/4), S, padding, phase_train, True, True)

            # Second convolution with BN and ReLU
            conv = self.convolution('Conv2', conv, F, int(K/4), 1, padding, phase_train, True, True)

            # Third layer without ReLU
            conv = self.convolution('Conv3', conv, 1, K, 1, padding, phase_train, True, False)

            # Downsample the residual input using strided 1x1 conv
            if S > 1: residual = self.convolution('Res_down', residual, 1, K, S, padding, phase_train, True, False)

            # Add the Residual
            conv = tf.add(conv, residual)
            conv = tf.nn.relu(conv, name=scope.name)

            # Get dimensions
            C = conv.get_shape().as_list()[-1]
            B = conv.get_shape().as_list()[0]

            # Apply channel wise dropout here
            if dropout and phase_train == True: conv = tf.nn.dropout(conv, dropout, noise_shape=[B, 1, 1, C])

            return conv


    def wide_residual_layer(self, scope, residual, K, S=2, padding='SAME', phase_train=None, dropout=None):

        """
        This is a wrapper for implementing a residual layer with inception layers
        :param scope:
        :param residual: Output of the previous layer
        :param K: Feature maps in the inception layer (will be multiplied by 4 during concatenation)
        :param S: Stride of the convolution, whether to downsample
        :param padding: SAME or VALID
        :param phase_train: For batch norm implementation
        :param dropout: keep prob for applying channel wise dropout
        :return:
        """

        # Set the scope. Implement a residual layer below
        with tf.variable_scope(scope) as scope:

            # First Convolution with BN and ReLU
            conv = self.inception_layer('Conv1', residual, K, S, padding, phase_train, True, True)

            # Second convolution without ReLU
            conv = self.inception_layer('Conv2', conv, K, 1, padding, phase_train, True, False)

            # Downsample the residual input using strided 1x1 conv
            if S > 1: residual = self.convolution('Res_down', residual, 1, K, S, padding, phase_train, True, False)

            # Add the Residual
            conv = tf.add(conv, residual)
            conv = tf.nn.relu(conv, name=scope.name)

            # Get dimensions
            C = conv.get_shape().as_list()[-1]
            B = conv.get_shape().as_list()[0]

            # Apply channel wise dropout here
            if dropout and phase_train == True: conv = tf.nn.dropout(conv, dropout, noise_shape=[B, 1, 1, C])

            return conv


    def residual_layer_stanford_3d(self, scope, X, F, K, S=2, K_prob=None, padding='SAME',
                       phase_train=None, DSC=False, BN=False, relu=False, dropout=None):
        """
        This is a wrapper for implementing a stanford style residual layer in 3 dimensions
        :param scope:
        :param X: Output of the previous layer, make sure this is not normalized and has no nonlinearity applied
        :param F: Dimensions of the second convolution in F(x) - the non inception layer one
        :param K: Feature maps in the inception layer (will be multiplied by 4 during concatenation)
        :param S: Stride of the convolution, whether to downsample
        :param k_prob: keep probability for dropout
        :param padding: SAME or VALID
        :param phase_train: For batch norm implementation
        :param DSC: Whether to perform standard downsample or incepted downsample
        :param BN: Whether to batch norm. Defaults to false to plug into another residual
        :param relu: whether to apply a nonlinearity at the end.
        :param dropout: keep prob for assigning channel wise dropout
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
            conv2 = self.convolution_3d('Conv2', conv1, F, K, 1, padding, phase_train, True, True)

            # Second dropout
            if K_prob: conv2 = tf.nn.dropout(conv2, K_prob)

            # Final downsampled conv without BN or RELU
            if DSC: conv = self.convolution_3d('ConvFinal', conv2, F, K, 1, padding, phase_train, False, False, True)
            else: conv = self.convolution_3d('ConvFinal', conv2, F, K*S, S, padding, phase_train, False, False)

            # Downsample the residual input if we did the conv layer. pool or strided
            if DSC: X = self.incepted_downsample_3d(X)
            elif S>1: X = self.convolution_3d('ResDown', X, 1, K*S, S, phase_train=phase_train, BN=False, relu=False)

            # Apply the batch normalization. Updates weights during training phase only
            if BN: conv = self.batch_normalization(conv, phase_train, 'BNRes')

            # The Residual block
            residual = tf.add(conv, X)

            # Relu activation
            if relu: residual = tf.nn.relu(residual, name=scope.name)

            # Get dimensions
            C = residual.get_shape().as_list()[-1]
            B = residual.get_shape().as_list()[0]

            # Apply channel wise dropout here
            if dropout and phase_train == True: residual = tf.nn.dropout(residual, dropout, noise_shape=[B, 1, 1, 1, C])

            return residual


    def residual_layer_3d(self, scope, residual, F, K, S=2, padding='SAME', phase_train=None, dropout=None):

        """
        This is a wrapper for implementing a microsoft style residual layer
        :param scope:
        :param residual: Output of the previous layer
        :param F: Dimensions of the second convolution in F(x) - the non inception layer one
        :param K: Feature maps in the inception layer (will be multiplied by 4 during concatenation)
        :param S: Stride of the convolution, whether to downsample
        :param padding: SAME or VALID
        :param phase_train: For batch norm implementation
        :param dropout: keep prob for applying channel wise dropout
        :return:
        """

        # Set the scope. Implement a residual layer below
        with tf.variable_scope(scope) as scope:

            # First Convolution with BN and ReLU
            conv = self.convolution_3d('Conv1', residual, F, K, S, padding, phase_train, True, True)

            # Second convolution without ReLU
            conv = self.convolution_3d('Conv2', conv, F, K, 1, padding, phase_train, True, False)

            # Downsample the residual input using strided 1x1 conv
            if S>1: residual = self.convolution_3d('Res_down', residual, 1, K, S, padding, phase_train, True, False)

            # Add the Residual
            conv = tf.add(conv, residual)
            conv = tf.nn.relu(conv, name=scope.name)

            # Get dimensions
            C = conv.get_shape().as_list()[-1]
            B = conv.get_shape().as_list()[0]

            # Apply channel wise dropout here
            if dropout and phase_train == True: conv = tf.nn.dropout(conv, dropout, noise_shape=[B, 1, 1, 1, C])

            return conv


    def residual_bottleneck_layer_3d(self, scope, residual, F, K, S=2, padding='SAME', phase_train=None, dropout=None):

        """
        This is a wrapper for implementing a microsoft style residual layer with bottlenecks for deep networks
        :param scope:
        :param residual: Output of the previous layer
        :param F: Dimensions of the second convolution in F(x) - the non inception layer one
        :param K: Feature maps in the inception layer (will be multiplied by 4 during concatenation)
        :param S: Stride of the convolution, whether to downsample
        :param padding: SAME or VALID
        :param phase_train: For batch norm implementation
        :param dropout: keep prob for applying channel wise dropout
        :return:
        """

        # Set the scope. Implement a residual layer below
        with tf.variable_scope(scope) as scope:

            # First Convolution with BN and ReLU
            conv = self.convolution_3d('Conv1', residual, 1, int(K/4), S, padding, phase_train, True, True)

            # Second convolution with BN and ReLU
            conv = self.convolution_3d('Conv2', conv, F, int(K/4), 1, padding, phase_train, True, True)

            # Third layer without ReLU
            conv = self.convolution_3d('Conv3', conv, 1, K, 1, padding, phase_train, True, False)

            # Downsample the residual input using strided 1x1 conv
            if S > 1: residual = self.convolution_3d('Res_down', residual, 1, K, S, padding, phase_train, True, False)

            # Add the Residual
            conv = tf.add(conv, residual)
            conv = tf.nn.relu(conv, name=scope.name)

            # Get dimensions
            C = conv.get_shape().as_list()[-1]
            B = conv.get_shape().as_list()[0]

            # Apply channel wise dropout here
            if dropout and phase_train == True: conv = tf.nn.dropout(conv, dropout, noise_shape=[B, 1, 1, 1, C])

            return conv


    def res_inc_layer(self, scope, X, K, S=2, K_prob=None, padding='SAME',
                       phase_train=None, DSC=False, BN=False, relu=False):
        """
        This is a wrapper for implementing a residual layer with incepted connections
        :param scope:
        :param X: Output of the previous layer, make sure this is not normalized and has no nonlinearity applied
        :param K: Feature maps in the inception layer (will be multiplied by 4 during concatenation)
        :param S: Stride of the convolution, whether to downsample
        :param k_prob: keep probability for dropout
        :param padding: SAME or VALID
        :param phase_train: For batch norm implementation
        :param DSC: Whether to perform standard downsample or incepted downsample
        :param BN: Whether to batch norm. Defaults to false to plug into another residual
        :param relu: whether to apply a nonlinearity at the end.
        :return:
        """

        # Set the scope. Implement a residual layer below
        with tf.variable_scope(scope) as scope:

            # Input is linear. Batch norm it then run it through a nonlinearity
            conv1 = self.batch_normalization(X, phase_train, scope)

            # ReLU
            conv1 = tf.nn.relu(conv1, scope.name)

            # Dropout
            if K_prob: conv1 = tf.nn.dropout(conv1, K_prob)

            # Another Convolution with BN and ReLu
            conv2 = self.inception_layer('Conv2', conv1, K, 1, padding, phase_train, True, True)

            # Second dropout
            if K_prob: conv2 = tf.nn.dropout(conv2, K_prob)

            # Final downsampled conv without BN or RELU
            if DSC:
                conv = self.inception_layer('ConvFinal', conv2, K, 1, padding, phase_train, False, False)
                conv = self.incepted_downsample(conv)
            else: conv = self.inception_layer('ConvFinal', conv2, K*S, S, padding, phase_train, False, False)

            # Downsample the residual input if we did the conv layer. pool or strided
            if DSC: X = self.incepted_downsample(X)
            elif S>1: X = self.convolution('ResDown', X, 2, K*S, S, phase_train=phase_train, BN=False, relu=False)

            # The Residual block
            residual = tf.add(conv, X)

            # Apply the batch normalization. Updates weights during training phase only
            if BN: residual = self.batch_normalization(residual, phase_train, scope)

            # Relu activation
            if relu: residual = tf.nn.relu(residual, name=scope.name)

            return residual


    def res_inc_layer_3d(self, scope, X, Fz, K, S=2, K_prob=None, padding='SAME',
                       phase_train=None, DSC=False, BN=False, relu=False):
        """
        This is a wrapper for implementing a residual layer with incepted layers between
        :param scope:
        :param X: Output of the previous layer, make sure this is not normalized and has no nonlinearity applied
        :param Fz: The z dimension of the inception filter
        :param K: Feature maps in the inception layer (will be multiplied by 4 during concatenation)
        :param S: Stride of the convolution, whether to downsample
        :param k_prob: keep probability for dropout
        :param padding: SAME or VALID
        :param phase_train: For batch norm implementation
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
            conv2 = self.inception_layer_3d('Inc1', conv1, K, Fz, 1, padding, phase_train, True, True)

            # Second dropout
            if K_prob: conv2 = tf.nn.dropout(conv2, K_prob)

            # Final downsampled conv without BN or RELU
            if DSC:
                conv = self.inception_layer_3d('IncFinal', conv2, K, Fz, 1, padding, phase_train, False, False)
                conv = self.incepted_downsample_3d(conv)
            else: conv = self.inception_layer_3d('IncFinal', conv2, K*S, Fz, S, padding, phase_train, False, False)

            # Downsample the residual input if we did the conv layer. pool or strided
            if DSC: X = self.incepted_downsample_3d(X)
            elif S>1: X = self.convolution_3d('ResDown', X, 2, K*S, S, phase_train=phase_train, BN=False, relu=False)

            # The Residual block
            residual = tf.add(conv, X)

            # Apply the batch normalization. Updates weights during training phase only
            if BN: residual = self.batch_normalization(residual, phase_train, 'BNRes')

            # Relu activation
            if relu: residual = tf.nn.relu(residual, name=scope.name)

            return residual


    # ***************  Miscellaneous layers ***************


    def fc7_layer(self, scope, X, neurons, dropout=False, phase_train=True, keep_prob=0.5, BN=False, relu=True, override=None, pad='VALID'):

        """
        Wrapper for implementing an FC layer based on a conv layer
        :param scope: Scopename of the layer
        :param X: Input of the prior layer
        :param neurons: Desired number of neurons in the layer
        :param dropout: Whether to implement dropout here
        :param phase_train: Are we in testing or training phase = only relevant for dropout
        :param keep_prob: if doing dropout, the keep probability
        :param BN: Batch norm or not
        :param relu: relu or not
        :param override: to override conv dimensions, if you want to average activations
        :return: result of all of the above. Averaged if overriden
        """

        # The Fc7 layer scope
        with tf.variable_scope(scope) as scope:

            # Retreive the size of the last layer
            batch_size, height, width, channel = X.get_shape().as_list()
            if override: height, width = override, override

            # Initialize the weights
            weights = tf.get_variable('weights', shape=[height * width * channel, neurons])

            # Define the biases
            bias = tf.get_variable('Bias', shape=[neurons], initializer=tf.constant_initializer(0.0))

            # Add to the weights collection
            tf.add_to_collection('weights', weights)
            tf.add_to_collection('biases', bias)

            # Reshape weights
            reshape = tf.reshape(weights, shape=[height, width, channel, neurons])

            # Convolution
            conv = tf.nn.conv2d(input=X, filter=reshape, strides=[1, 1, 1, 1], padding=pad, name='Conv')

            # Optional batch norm
            if BN: conv = self.batch_normalization(conv, phase_train, 'Fc7Norm')

            # Add biases
            conv = tf.nn.bias_add(conv, bias)

            # Optional relu
            if relu: conv = tf.nn.relu(conv, name=scope.name)

            # Dropout here if wanted and in train phase
            if (phase_train == True) and dropout: conv = tf.nn.dropout(conv, keep_prob)

            # Average outputs if we used an override
            if override:
                F = conv.get_shape().as_list()[1]
                conv = tf.nn.avg_pool(conv, [1, F, F, 1], [1, 2, 2, 1], 'VALID')

            # Activation summary
            if self.summary: self._activation_summary(conv)

            return tf.squeeze(conv)


    def fc7_layer_3d(self, scope, X, neurons, dropout=False, phase_train=True, keep_prob=0.5,
                  BN=False, relu=True, override=None, pad='VALID'):

        """
        Wrapper for implementing a 3D FC layer based on a conv layer
        :param scope: Scopename of the layer
        :param X: Input of the prior layer
        :param neurons: Desired number of neurons in the layer
        :param dropout: Whether to implement dropout here
        :param phase_train: Are we in testing or training phase = only relevant for dropout
        :param keep_prob: if doing dropout, the keep probability
        :param BN: Batch norm or not
        :param relu: relu or not
        :param override: to override conv dimensions, if you want to average activations
        :return: result of all of the above. Averaged if overriden
        """

        # The Fc7 layer scope
        with tf.variable_scope(scope) as scope:

            # Retreive the size of the last layer
            batch_size, height, width, depth, channel = X.get_shape().as_list()
            if override: height, width , depth = override, override, override

            # Initialize the weights
            weights = tf.get_variable('weights', shape=[height * width * depth * channel, neurons])

            # Define the biases
            bias = tf.get_variable('Bias', shape=[neurons], initializer=tf.constant_initializer(0.0))

            # Add to the weights collection
            tf.add_to_collection('weights', weights)
            tf.add_to_collection('biases', bias)

            # Reshape weights
            reshape = tf.reshape(weights, shape=[height, width, depth, channel, neurons])

            # Convolution
            conv = tf.nn.conv3d(input=X, filter=reshape, strides=[1, 1, 1, 1, 1], padding=pad, name='Conv')

            # Optional batch norm
            if BN: conv = self.batch_normalization(conv, phase_train, 'Fc7Norm')

            # Add biases
            conv = tf.nn.bias_add(conv, bias, name='Bias')

            # Optional relu
            if relu: conv = tf.nn.relu(conv, name=scope.name)

            # Dropout here if wanted and in train phase
            if (phase_train == True) and dropout: conv = tf.nn.dropout(conv, keep_prob)

            # Average outputs if we used an override
            if override:
                F = conv.get_shape().as_list()[1]
                conv = tf.nn.avg_pool3d(conv, [1, F, F, F, 1], [1, 2, 2, 2, 1], 'VALID')

            # Activation summary
            if self.summary: self._activation_summary(conv)

            return tf.squeeze(conv)


    def fc7_layer_old(self, scope, X, neurons, dropout=False, phase_train=True, keep_prob=0.5, BN=False):
        """
        Wrapper for implementing a fully connected layer
        :param scope: Scopename of the layer
        :param X: Input of the prior layer
        :param neurons: Desired number of neurons in the layer
        :param dropout: Whether to implement dropout here
        :param phase_train: Are we in testing or training phase = only relevant for dropout
        :param keep_prob: if doing dropout, the keep probability
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
                                      initializer=tf.contrib.layers.variance_scaling_initializer())

            # Add to the collection of weights
            tf.add_to_collection('weights', weights)

            # Initialize the biases
            biases = tf.Variable(np.ones(neurons), name='Bias', dtype=tf.float32)

            # Do the math
            fc7 = tf.matmul(reshape, weights)
            if BN: fc7 = self.batch_normalization(fc7, phase_train, 'Fc7Norm')
            fc7 = tf.nn.relu(fc7 + biases, name=scope.name)

            # Dropout here if wanted and in train phase
            if (phase_train == True) and dropout: fc7 = tf.nn.dropout(fc7, keep_prob)

            # Activation summary
            if self.summary: self._activation_summary(fc7)

            return fc7


    def linear_layer(self, scope, X, neurons, dropout=False, phase_train=True, keep_prob=0.5, BN=False, relu=True, add_bias=True):
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
        :return:
        """

        # Retreive the size of the last layer
        dim = X.get_shape().as_list()[-1]

        # The linear layer Dimensions:
        with tf.variable_scope(scope) as scope:

            # Initialize the weights
            weights = tf.get_variable('weights', shape=[dim, neurons],
                                      initializer=tf.contrib.layers.variance_scaling_initializer())

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
                                 initializer=tf.contrib.layers.variance_scaling_initializer())
            B1 = tf.get_variable('Bias1', shape=[20], initializer=tf.contrib.layers.variance_scaling_initializer())
            W2 = tf.get_variable('Weights2', shape=[20, 6], initializer=tf.contrib.layers.variance_scaling_initializer())

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


    def transitional_layer(self, scope, X, K, S=1, padding='SAME', phase_train=None, BN=True, relu=True):
        """
        This function implements a transition layer to insert before the FC layer. Improves regularization
        :param scope:
        :param X: Output of the previous layer
        :param K: Feature maps in the inception layer (will be multiplied by 4 during concatenation)
        :param S: Stride Whether to downsample the convoluted inceptions
        :param padding:
        :param phase_train: For batch norm implementation
        :param BN: whether to perform batch normalization
        :return: the inception layer output after concat
        """

        # Implement an incepted transition layer here ----------------
        with tf.variable_scope(scope) as scope:

            # Retreive size of prior network. Prior = [batch, F, F, K]
            F = X.get_shape().as_list()[1]

            # First branch, 7x7 conv then global avg pool
            inception1a = self.convolution('Transition1', X, 7, K, S, padding, phase_train, BN, relu)
            inception1 = tf.nn.avg_pool(inception1a, [1, F, F, 1], [1, 2, 2, 1], 'VALID')

            # Second branch, 5x5 conv then global avg pool
            inception2a = self.convolution('Transition2', X, 5, K, S, padding, phase_train, BN, relu)
            inception2 = tf.nn.avg_pool(inception2a, [1, F, F, 1], [1, 2, 2, 1], 'VALID')

            # Third branch, 3x3 conv then global avg pool
            inception3a = self.convolution('Transition3', X, 3, K, S, padding, phase_train, BN, relu)
            inception3 = tf.nn.avg_pool(inception3a, [1, F, F, 1], [1, 2, 2, 1], 'VALID')

            # Concatenate the results
            inception = tf.concat([inception1, inception2, inception3], axis=-1)

            return tf.squeeze(inception)


    def incepted_downsample(self, X, S=2, padding='SAME'):
        """
        This function implements a downsampling layer that utilizes both an average and max pooling operation
        :param scope:
        :param X: Output of the previous layer
        :param S: The degree of downsampling or stride
        :param padding: SAME or VALID
        :return: the layer output after concat
        """

        # 1st branch, AVG pool
        avg = tf.nn.avg_pool(X, [1, 2, 2, 1], [1, S, S, 1], padding)

        # 2nd branch, max pool
        maxi = tf.nn.max_pool(X, [1, 2, 2, 1], [1, S, S, 1], padding)

        # Concatenate the results
        inception = tf.concat([avg, maxi], -1)

        # Create a histogram/scalar summary of the conv1 layer
        if self.summary: self._activation_summary(inception)

        return inception


    def incepted_downsample_3d(self, X, S=2, padding='SAME'):
        """
        This function implements a 3d downsampling layer that utilizes both an average and max pooling operation
        :param scope:
        :param X: Output of the previous layer
        :param S: The degree of downsampling or stride
        :param padding: SAME or VALID
        :return: the layer output after concat
        """

        # 1st branch, AVG pool
        avg = tf.nn.avg_pool3d(X, [1, 2, 2, 2, 1], [1, S, S, S, 1], padding)

        # 2nd branch, max pool
        maxi = tf.nn.max_pool3d(X, [1, 2, 2, 2, 1], [1, S, S, S, 1], padding)

        # Concatenate the results
        inception = tf.concat([avg, maxi], -1)

        # Create a histogram/scalar summary of the conv1 layer
        if self.summary: self._activation_summary(inception)

        return inception


    def global_avg_pool(self, X, S=1, padding='VALID'):

        """
        Implements global average pooling
        :param X: input layer
        :param S: stride of pool
        :param padding: if you want to pad for whatever reason
        :return: output of dimension 1
        """

        # First retreive dimensions of input
        Fx = X.get_shape().as_list()[1]
        Fy = X.get_shape().as_list()[2]

        # Now perform pool operation
        return tf.nn.avg_pool(X, [1, Fx, Fy, 1], [1, S, S, 1], padding)


    def global_avg_pool3D(self, X, S=1, padding='VALID'):

        """
        Implements global average pooling
        :param X: input layer
        :param S: stride of pool
        :param padding: if you want to pad for whatever reason
        :return: output of dimension 1
        """

        # First retreive dimensions of input
        Fx = X.get_shape().as_list()[1]
        Fy = X.get_shape().as_list()[2]
        Fz = X.get_shape().as_list()[3]

        # Now perform pool operation
        return tf.nn.avg_pool3d(X, [1, Fx, Fy, Fz, 1], [1, S, S, S, 1], padding)



    # ***************  Loss function wrappers ***************



    def segmentation_SCE_loss(self, logits, labelz, class_factor=1.2):
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
        if self.summary: tf.summary.scalar('loss', loss)

        # Add these losses to the collection
        tf.add_to_collection('losses', loss)

        return loss


    def MSE_loss(self, logits, labels, mask_factor=0.0, mask_avg=0.0, mask_norm=0.0, debug=False):
        """
        Calculates the mean squared error for linear regression
        :param logits: network outputs
        :param labels: actual values
        :param mask_factor:
        :param mask_avg:
        :param mask_norm:
        :param summary:
        :return: The loss value
        """

        # Must squeeze because otherwise we may subtract a row vector from a column vector 9giving a matrix)
        labels = tf.squeeze(labels)
        logits = tf.squeeze(logits)

        # For distance sensitive mask based on distance from given mask avg
        mask = tf.cast(labels, tf.float32)

        # Now normalize so that something one norm away gets 2
        mask = tf.add(tf.multiply(tf.divide(tf.abs(tf.subtract(mask_avg, mask)), mask_norm), mask_factor), 1.0)

        # Convert to row vector
        mask = tf.squeeze(mask)

        # Print debug summary for possible dimensionality issues
        if debug: print (labels, logits, labels-logits)

        # Calculate MSE with the factor multiplied in
        if mask_factor: MSE_loss = tf.reduce_mean(tf.multiply(tf.square(labels - logits), mask))
        else: MSE_loss = tf.reduce_mean(tf.square(labels - logits))

        # Output the summary of the MSE and MAE
        if self.summary:
            tf.summary.scalar('Square Error', MSE_loss)
            tf.summary.scalar('Absolute Error', tf.reduce_mean(tf.abs(labels - logits)))

        # Add these losses to the collection
        tf.add_to_collection('losses', MSE_loss)

        # For now return MSE loss, add L2 regularization below later
        return MSE_loss


    def SCE_loss(self, logits, labels, num_classes):
        """
        Calculates the softmax cross entropy loss between logits and labels
        :param logits:
        :param labels:
        :param num_classes: the number of classes
        :return:
        """
        # Change labels to one hot
        labels = tf.one_hot(tf.cast(labels, tf.uint8), depth=num_classes, dtype=tf.uint8)

        # Calculate  loss
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        # Reduce to scalar
        loss = tf.reduce_mean(loss)

        # Output the summary of the MSE and MAE
        if self.summary: tf.summary.scalar('Cross Entropy', loss)

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


    def calc_L2_Loss(self, L2_gamma):
        """
        Calculates the L2 Loss
        :param L2_gamma:
        :return:
        """

        # Retreive the weights collection
        weights = tf.get_collection('weights')

        # Sum the losses
        L2_loss = tf.multiply(tf.add_n([tf.nn.l2_loss(v) for v in weights]), L2_gamma)

        # Add it to the collection
        tf.add_to_collection('losses', L2_loss)

        # Activation summary
        if self.summary: tf.summary.scalar('L2_Loss', L2_loss)

        return L2_loss


    def pixel_wise_softmax(self, output_map):
        """
        calculates a pixel by pixel softmax score from a networks output map
        :param output_map: the output logits of a unet for example
        :return: the softmax pixel by pixel
        """

        # Exponentiate the values
        exponential_map = tf.exp(output_map)

        # Create the denominator
        sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
        tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))

        # return the result of the softmax
        return tf.div(exponential_map, tensor_sum_exp)


    def DICE_loss(self, logitz, labelz, num_classes=2, network_dims=64):

        """
        Cost function
        :param logitz: The raw log odds units output from the network
        :param labelz: The labels: not one hot encoded
        :param num_classes: number of classes predicted
        :param class_weights: class weight array
        :param loss_type: DICE or other to use dice or weighted
        :return:
        """

        # Reduce dimensionality
        labelz = tf.squeeze(labelz)

        # Remove background label
        labels = tf.cast(labelz > 1, tf.uint8)

        # Summary images
        if self.summary:
            tf.summary.image('Labels', tf.reshape(tf.cast(labels[2], tf.float32), shape=[1, network_dims, network_dims, 1]), 2)
            tf.summary.image('Logits', tf.reshape(logitz[2, :, :, 1], shape=[1, network_dims, network_dims, 1]), 2)

        # Make labels one hot
        labels = tf.cast(tf.one_hot(labels, depth=2, dtype=tf.uint8), tf.float32)

        # Generate mask
        mask = tf.expand_dims(tf.cast(labelz > 0, tf.float32), -1)

        # Apply mask
        logits, labels = logitz * mask, labels * mask

        # Flatten
        logits = tf.reshape(logits, [-1, num_classes])
        labels = tf.reshape(labels, [-1, num_classes])

        # To prevent number errors:
        eps = 1e-5

        # Calculate softmax:
        logits = tf.nn.softmax(logits)

        # Find the intersection
        intersection = 2 * tf.reduce_sum(logits * labels)

        # find the union
        union = eps + tf.reduce_sum(logits) + tf.reduce_sum(labels)

        # Calculate the loss
        dice = intersection / union

        # Output the training DICE score
        tf.summary.scalar('DICE_Score', dice)

        # 1-DICE since we want better scores to have lower loss
        loss = 1 - dice

        # Output the Loss
        tf.summary.scalar('Loss_Raw', loss)

        # Add these losses to the collection
        tf.add_to_collection('losses', loss)

        return loss


    # *************** Utility Functions ***************


    def batch_normalization(self, conv, phase_train, scope):

        """
        Wrapper for the batch normalization implementation
        :param conv: The layer to normalize
        :param phase_train: Is this training or testing phase
        :param scope: the scope for this operation
        :return: conv: the normalized convolution
        """

        return tf.layers.batch_normalization(conv, training=phase_train, momentum=0.98)


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


class DenseNet(SODMatrix):

    # Variables constant to all instances here

    def __init__(self, nb_blocks, filters, sess, phase_train, summary):

        # Variables accessible to only specific instances here:
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.sess = sess
        self.phase_train = phase_train
        self.summary = summary


    def bottleneck_layer(self, X, scope, keep_prob=None):
        """
        Implements a bottleneck layer with BN-->ReLU -> 1x1 Conv -->BN/ReLU --> 3x3 conv
        :param x:  input
        :param scope: scope of the operations
        :param phase_train: whether in training or testing
        :return: results
        """

        with tf.name_scope(scope):

            # Batch norm first
            conv = self.batch_normalization(X, self.phase_train, scope)

            # ReLU
            conv = tf.nn.relu(conv)

            # 1x1 conv: note BN and Relu applied by default
            conv = self.convolution(scope, conv, 1, self.filters, 1, 'SAME', self.phase_train)

            # Dropout (note this is after BN and relu in this case)
            if keep_prob and self.phase_train==True: conv = tf.nn.dropout(conv, keep_prob)

            # 3x3 conv, don't apply BN and relu
            conv = self.convolution(scope+'_2', conv, 3, self.filters, 1, 'SAME', self.phase_train, BN=False, relu=False)

            # Dropout (note that this is before BN and relu)
            if keep_prob and self.phase_train == True: conv = tf.nn.dropout(conv, keep_prob)

            return conv


    def bottleneck_layer_3d(self, X, scope, keep_prob=None):
        """
        Implements a 3D bottleneck layer with BN-->ReLU -> 1x1 Conv -->BN/ReLU --> 3x3 conv
        :param x:  input
        :param scope: scope of the operations
        :param phase_train: whether in training or testing
        :return: results
        """

        with tf.name_scope(scope):

            # Batch norm first
            conv = self.batch_normalization(X, self.phase_train, None)

            # ReLU
            conv = tf.nn.relu(conv)

            # 1x1 conv: note BN and Relu applied by default
            conv = self.convolution_3d(scope, conv, [1, 1, 1], self.filters, 1, 'SAME', self.phase_train)

            # Dropout (note this is after BN and relu in this case)
            if keep_prob and self.phase_train==True: conv = tf.nn.dropout(conv, keep_prob)

            # 3x3 conv, don't apply BN and relu
            conv = self.convolution_3d(scope+'_2', conv, [1, 3, 3], self.filters, 1, 'SAME', self.phase_train, BN=False, relu=False)

            # Dropout (note that this is before BN and relu)
            if keep_prob and self.phase_train == True: conv = tf.nn.dropout(conv, keep_prob)

            return conv


    def transition_layer(self, X, scope, keep_prob=None):

        """
        Transition layer for Densenet: not wide
        :param X: input
        :param scope: scope
        :param phase_train:
        :return:
        """

        with tf.name_scope(scope):

            # BN first
            conv = self.batch_normalization(X, self.phase_train, scope)

            # ReLU
            conv = tf.nn.relu(conv)

            # Conv 1x1
            conv = self.convolution(scope, conv, 1, self.filters, 1, 'SAME', self.phase_train, BN=False, relu=False)

            # Dropout
            if keep_prob and self.phase_train == True: conv = tf.nn.dropout(conv, keep_prob)

            # Average pool
            conv = tf.nn.avg_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

            return conv


    def transition_layer_3d(self, X, scope):

        """
        3D Transition layer for Densenet: Uses strided convolutions
        :param X: input
        :param scope: scope
        :return:
        """

        with tf.name_scope(scope):

            # ReLu and BN first since we're using a convolution to downsample
            conv = tf.nn.relu(self.batch_normalization(X, self.phase_train, None))

            # First downsample the Z axis - double filters to prevent bottleneck
            conv = self.convolution_3d(scope+'_down', conv, [2, 1, 1], 2*self.filters, 1, 'VALID', self.phase_train, BN=False, relu=False)

            # Now do the average pool to downsample the X and Y planes
            conv = tf.nn.avg_pool3d(conv, [1, 1, 2, 2, 1], [1, 1, 2, 2, 1], 'SAME')

            return conv


    def dense_block(self, input_x, nb_layers, layer_name, keep_prob=None, downsample=False):

        """
        Creates a dense block connection all same sized filters
        :param input_x: The input to this dense block (output of prior downsample operation)
        :param nb_layers: how many layers desired
        :param layer_name: base name of this block
        :param keep_prob: Whether to use dropout
        :param trans: whether to include a downsample at the end
        :return:
        """

        with tf.name_scope(layer_name):

            # Array to hold each layer
            layers_concat = []

            # Append to list
            layers_concat.append(input_x)

            # The first layer of this block
            conv = self.bottleneck_layer(input_x, (layer_name+'_denseN_0'), keep_prob)

            # Concat the first layer
            layers_concat.append(conv)

            # Loop through the number of layers desired
            for z in range(nb_layers):

                # Concat all the prior layer into this layer
                conv = tf.concat(layers_concat, axis=-1)

                # Create a new layer
                conv = self.bottleneck_layer(conv, (layer_name+'_denseN_'+str(z+1)), keep_prob)

                # Append this layer to the running list of dense connected layers
                layers_concat.append(conv)

            # Combine the layers
            conv = tf.concat(layers_concat, axis=-1)

            # Downsample if requested
            if downsample: conv = self.transition_layer(conv, (layer_name+'_Downsample'), keep_prob)

            return conv


class DenseUnet(DenseNet):

    """
    Class for creating unets Dense style
    """

    def __init__(self, nb_blocks, filters, images, phase_train):

        # Variables accessible to only specific instances here:
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.phase_train = phase_train
        self.images = images


    def dense_block(self, input_x, nb_layers, layer_name, keep_prob=None, downsample=False):

        """
        Creates a dense block
        :param input_x: The input to this dense block (output of prior downsample operation)
        :param nb_layers: how many layers desired
        :param layer_name: base name of this block
        :param keep_prob: Whether to use dropout
        :param trans: whether to include a downsample at the end
        :return:
        """

        with tf.name_scope(layer_name):

            # Array to hold each layer
            layers_concat = []

            # Append to list
            layers_concat.append(input_x)

            # The first layer of this block
            conv = self.bottleneck_layer(input_x, (layer_name+'_denseN_0'), keep_prob)

            # Concat the first layer
            layers_concat.append(conv)

            # Loop through the number of layers desired
            for z in range(nb_layers):

                # Concat all the prior layer into this layer
                conv = tf.concat(layers_concat, axis=-1)

                # Create a new layer
                conv = self.bottleneck_layer(conv, (layer_name+'_denseN_'+str(z+1)), keep_prob)

                # Append this layer to the running list of dense connected layers
                layers_concat.append(conv)

            # Combine the layers
            conv = tf.concat(layers_concat, axis=-1)

            if downsample: conv = self.transition_layer(conv, (layer_name+'_Downsample'), keep_prob)

            return conv, layers_concat


    def dense_block_3d(self, input_x, nb_layers, layer_name, keep_prob=None):

        """
        Creates a dense block
        :param input_x: The input to this dense block (output of prior downsample operation)
        :param nb_layers: how many layers desired
        :param layer_name: base name of this block
        :param keep_prob: Whether to use dropout
        :return:
        """

        with tf.name_scope(layer_name):

            # Array to hold each layer
            layers_concat = []

            # Append to list
            layers_concat.append(input_x)

            # The first layer of this block
            conv = self.bottleneck_layer_3d(input_x, (layer_name+'_denseN_0'), keep_prob)

            # Concat the first layer
            layers_concat.append(conv)

            # Loop through the number of layers desired
            for z in range(nb_layers):

                # Concat all the prior layer into this layer
                conv = tf.concat(layers_concat, axis=-1)

                # Create a new layer
                conv = self.bottleneck_layer_3d(conv, (layer_name+'_denseN_'+str(z+1)), keep_prob)

                # Append this layer to the running list of dense connected layers
                layers_concat.append(conv)

            # The concat has to be 2D, first retreive the Z and K size
            concat = tf.concat(layers_concat, -1)
            Fz, Ch = concat.get_shape().as_list()[1], concat.get_shape().as_list()[-1]

            # Now create a projection matrix
            concat = self.convolution_3d(layer_name+'_projection', concat, [Fz, 1, 1], Ch, 1, 'VALID', self.phase_train, BN=False)

            return conv, tf.squeeze(concat)


    def up_transition(self, scope, X, F, K, S, concat_var=None, padding='SAME'):

        """
        Deconvolutions for the DenseUnet
        :param scope:
        :param X:
        :param F:
        :param K:
        :param S:
        :param padding:
        :param concat_var:
        :param summary:
        :return:
        """

        with tf.variable_scope(scope) as scope:

            # Set channel size based on input depth
            C = X.get_shape().as_list()[-1]

            # He init
            kernel = tf.get_variable('Weights', shape=[F, F, K, C], initializer=tf.contrib.layers.variance_scaling_initializer())

            # Define the biases
            bias = tf.get_variable('Bias', shape=[K], initializer=tf.constant_initializer(0.0))

            # Add to the weights collection
            tf.add_to_collection('weights', kernel)
            tf.add_to_collection('biases', bias)

            # Define the output shape
            out_shape = X.get_shape().as_list()
            out_shape[1] *= 2
            out_shape[2] *= 2
            out_shape[3] = K

            # Perform the deconvolution. output_shape: A 1-D Tensor representing the output shape of the deconvolution op.
            conv = tf.nn.conv2d_transpose(X, kernel, output_shape=out_shape, strides=[1, S, S, 1], padding=padding)

            # Add in bias
            conv = tf.nn.bias_add(conv, bias)

            # Concatenate
            conv = tf.concat([concat_var, conv], axis=-1)

            # Create a histogram summary and summary of sparsity
            if self.summary: self._activation_summary(conv)

            return conv


    def up_transition_3d(self, scope, X, F, K, S, concat_var=None, padding='VALID'):

        """
        Deconvolutions for the DenseUnet
        :param scope:
        :param X:
        :param F:
        :param K:
        :param S:
        :param padding:
        :param concat_var:
        :param summary:
        :return:
        """

        with tf.variable_scope(scope) as scope:

            # Set channel size based on input depth
            C = X.get_shape().as_list()[-1]

            # He init
            kernel = tf.get_variable('Weights', shape=[F, F, F, K, C], initializer=tf.contrib.layers.variance_scaling_initializer())

            # Define the biases
            bias = tf.get_variable('Bias', shape=[K], initializer=tf.constant_initializer(0.0))

            # Add to the weights collection
            tf.add_to_collection('weights', kernel)
            tf.add_to_collection('biases', bias)

            # Define the output shape
            out_shape = X.get_shape().as_list()
            out_shape[1] *= 2
            out_shape[2] *= 2
            out_shape[3] *= 2
            out_shape[3] = K

            # Perform the deconvolution. output_shape: A 1-D Tensor representing the output shape of the deconvolution op.
            conv = tf.nn.conv2d_transpose(X, kernel, output_shape=out_shape, strides=[1, S, S, S, 1], padding=padding)

            # Add in bias
            conv = tf.nn.bias_add(conv, bias)

            # Concatenate
            conv = tf.concat([concat_var, conv], axis=-1)

            # Create a histogram summary and summary of sparsity
            if self.summary: self._activation_summary(conv)

            return conv


    def define_network(self, layers = [], keep_prob=None):

        # conv holds output of bottleneck layers (no BN/ReLU). Concat holds running lists of skip connections
        conv, concat = [None] * (self.nb_blocks + 1), [None] * (self.nb_blocks - 1)

        # Define the first layers before starting the Dense blocks
        conv[0] = self.convolution('Conv1', self.images, 3, 2*self.filters, 1, phase_train=self.phase_train, BN=False, relu=False)

        # Loop through and make the downsample blocks
        for z in range (self.nb_blocks):

            # Z holds the prior index, X holds the layer index
            x = z+1

            # Generate a dense block
            conv[x], concat[z] = self.dense_block(conv[z], layers[z], 'Dense_'+str(x), keep_prob)

            # Downsample unless at the end
            if x < self.nb_blocks: conv[x] = self.transition_layer(conv[x], 'Downsample_'+str(x), keep_prob)

        # Set first dconv to output of final conv
        deconv = tf.nn.relu(self.batch_normalization(conv[-1], self.phase_train, None))

        # Now loop through and perform the upsamples. No longer need to store these
        for z in range(self.nb_blocks):

            # Z holds the prior index, X holds the layer index
            x = z+1

            # Perform upsample unless at the end
            if x < self.nb_blocks: deconv = self.up_transition('Upsample_'+str(x), deconv, 3, self.filters, 2, concat[-x])

            # Generate a dense block
            deconv, _ = self.dense_block(deconv, layers[-(z+2)], 'Up_Dense_' + str(x), keep_prob)

        # Return final layer after batch norm and relu
        return tf.nn.relu(self.batch_normalization(deconv, self.phase_train, 'BNa'))


    def define_network_25D(self, layers = [], keep_prob=None, prints=True):

        # conv holds output of bottleneck layers (no BN/ReLU). Concat holds running lists of skip connections
        conv, concat = [None]*(self.nb_blocks+1), [None]*(self.nb_blocks-1)

        # Define the first layers before starting the Dense blocks
        conv[0] = self.convolution_3d('Conv1', self.images, [1, 3, 3], 2*self.filters, 1, 'SAME', self.phase_train, BN=False, relu=False)

        # Loop through and make the downsample blocks
        for z in range (self.nb_blocks):

            # Z holds the prior index, X holds the layer index
            x = z+1

            # Retreive number of slices here
            slices = conv[z].get_shape().as_list()[1]

            # Error handling for when we are down to rank 4
            if len(conv[z].get_shape().as_list()) == 4: slices=1

            # Generate a dense block. 3D if slices > 1
            if slices > 1: conv[x], concat[z] = self.dense_block_3d(conv[z], layers[z], 'Dense_'+str(x), keep_prob)
            else: conv[x], _ = self.dense_block(tf.squeeze(conv[z]), layers[z], 'Dense_'+str(x), keep_prob)

            # Downsample unless at the end
            if x < self.nb_blocks and slices > 1: conv[x] = self.transition_layer_3d(conv[x], 'Downsample_'+str(x))
            elif x < self.nb_blocks and slices ==1: conv[x] = self.transition_layer(conv[x], 'Downsample_'+str(x))

        # Set first dconv to output of final conv
        deconv = tf.nn.relu(self.batch_normalization(conv[-1], self.phase_train, None))

        # Now loop through and perform the upsamples. No longer need to store these. These are 2D
        for z in range(self.nb_blocks-1):

            # Z holds the prior index, X holds the layer index
            x = z+1

            # Perform upsample unless at the end
            if x < self.nb_blocks: deconv = self.up_transition('Upsample_'+str(x), deconv, 3, self.filters, 2, concat[-x])

            # Generate a dense block
            deconv, _ = self.dense_block(deconv, layers[-(z+2)], 'Up_Dense_' + str(x), keep_prob)

        # Return final layer after batch norm and relu
        return tf.nn.relu(self.batch_normalization(deconv, self.phase_train, None))


class ResNet(SODMatrix):


    def __init__(self, nb_blocks, filters, images, sess, phase_train, summary):

        # Variables accessible to only specific instances here:

        self.nb_blocks = nb_blocks
        self.images = images
        self.filters = filters
        self.sess = sess
        self.phase_train = phase_train
        self.summary = summary


    def residual_block(self, input_x, nb_layers, layer_name, K, F=3, padding='SAME', downsample=True, stanford=False):

        """
        Implements a block of residual layers at the same spatial dimension
        :param input_x: Input, either from the last conv layer or the images
        :param nb_layers: number of residual layers
        :param layer_name: the baseline name of this block
        :param K: feature map size
        :param F: filter size
        :param padding: SAME or VALID
        :param downsample: Whether to downsample at the end
        :param stanford: whether to use stanford style layers
        :return:
        """

        with tf.name_scope(layer_name):

            # The first layer of this block
            conv = self.residual_layer((layer_name+'_res_0'), input_x, F, K, 1, padding, self.phase_train)

            # Loop through the number of layers desired
            for z in range(nb_layers):

                # Perform the desired operations
                conv = self.residual_layer((layer_name+'_res_'+str(z+1)), conv, F, K, 1, padding, self.phase_train)

            # Downsample if requested
            if downsample:
                conv = self.residual_layer((layer_name+'_res_down_'), conv, F, K*2, 2, padding, self.phase_train)

            return conv


    def inception_block(self, input_x, nb_layers, layer_name, K, padding='SAME', downsample=True):

        """
        Implements a block of inception layers at the same spatial dimension
        :param input_x: Input, either from the last conv layer or the images
        :param nb_layers: number of layers
        :param layer_name: the baseline name of this block
        :param F: filter size
        :param padding: SAME or VALID
        :param downsample: Whether to downsample at the end
        :return:
        """

        with tf.name_scope(layer_name):

            # The first layer of this block
            conv = self.inception_layer((layer_name + '_inc_0'), input_x, K, 1, padding, self.phase_train)

            # Loop through the number of layers desired
            for z in range(nb_layers):

                # Perform the desired operations
                conv = self.inception_layer((layer_name + '_inc_' + str(z + 1)), conv, K, 1, padding, self.phase_train)

            # Downsample if requested
            if downsample:
                conv = self.inception_layer((layer_name + '_inc_down_'), conv, K * 2, 2, padding, self.phase_train)

            return conv


    def up_transition(self, scope, X, F, K, S, concat_var=None, padding='SAME', res=True):

        """
        Performs an upsampling procedure
        :param scope:
        :param X: Inputs
        :param F: Filter sizes
        :param K: Kernel sizes
        :param S: Stride size
        :param concat_var: The skip connection
        :param padding: SAME or VALID
        :param res: Whether to concatenate or add the skip connection
        :return:
        """

        with tf.variable_scope(scope) as scope:

            # Set channel size based on input depth
            C = X.get_shape().as_list()[-1]

            # He init
            kernel = tf.get_variable('Weights', shape=[F, F, K, C], initializer=tf.contrib.layers.variance_scaling_initializer())

            # Define the biases
            bias = tf.get_variable('Bias', shape=[K], initializer=tf.constant_initializer(0.0))

            # Add to the weights collection
            tf.add_to_collection('weights', kernel)
            tf.add_to_collection('biases', bias)

            # Define the output shape
            out_shape = X.get_shape().as_list()
            out_shape[1] *= 2
            out_shape[2] *= 2
            out_shape[3] = K

            # Perform the deconvolution. output_shape: A 1-D Tensor representing the output shape of the deconvolution op.
            conv = tf.nn.conv2d_transpose(X, kernel, output_shape=out_shape, strides=[1, S, S, 1], padding=padding)

            # Concatenate
            if res: conv = tf.add(conv, concat_var)
            else: conv = tf.concat([concat_var, conv], axis=-1)

            # Apply the batch normalization. Updates weights during training phase only
            conv = self.batch_normalization(conv, self.phase_train, scope)

            # Add in bias
            conv = tf.nn.bias_add(conv, bias)

            # Relu
            conv = tf.nn.relu(conv, name=scope.name)

            # Create a histogram summary and summary of sparsity
            if self.summary: self._activation_summary(conv)

            return conv


    def define_network(self, block_layers=[], inception_layers=[], F=3, S_1=1, padding='SAME', downsample_last=False, FPN=True, FPN_layers=64):

        """
        Shortcut to creating a residual or residual-inception style network with just a few lines of code
        :param input_images: The input images
        :param block_layers: How many layers in each downsample block. i.e.: [2, 4, 6, ...] for 2 layers, then 4 etc
        :param inception_layers: which block numbers to make inception layers: [0, 0, 1] makes the third block inception
        :param F: Filter sizes to use, default to 3
        :param S_1: Stride of the initial convolution: sometimes we don't want to downsample here
        :param padding: Padding to use, default ot 'SAME'
        :param downsample_last: Whether to downsample the last block
        :param: FPN = Whether to perform a FPN or UNet arm at the end and return the output activation maps
        :param FPN_layers: How many layers to output in the FPN
        :return:
                conv[-1]: the final conv layer
                conv: the array of outputs from each block: If S1 is 1, keep in mind each index return the result of the block
                    which is just the downsampled final layer. it RUNs at a higher feature map size
        """

        # conv holds output of bottleneck layers (no BN/ReLU). Concat holds running lists of skip connections
        conv = [None] * (self.nb_blocks + 1)

        # Define the first layers before starting the Dense blocks
        conv[0] = self.convolution('Conv1', self.images, F, self.filters, S_1, phase_train=self.phase_train)

        # To save filters for later
        filter_size_buffer = []

        # Loop through and make the downsample blocks
        for z in range (self.nb_blocks):

            # Z holds the prior index, X holds the current layer index
            x = z+1

            # Set filter size for this block
            if S_1 == 1: filters = self.filters * (2**z)
            else: filters = self.filters * (2**x)
            filter_size_buffer.append(filters)

            # Generate the appropriate block, only downsample if not at the end
            if inception_layers[z]:
                if (not downsample_last) and x==self.nb_blocks: conv[x] = self.inception_block(conv[z], block_layers[z], 'Inc_'+str(x), filters, padding, False)
                else: conv[x] = self.inception_block(conv[z], block_layers[z], 'Inc_'+str(x), filters, padding, True)

            else:
                if (not downsample_last) and x==self.nb_blocks: conv[x] = self.residual_block(conv[z], block_layers[z], 'Res_'+str(x), filters, F, padding, False, False)
                else: conv[x] = self.residual_block(conv[z], block_layers[z], 'Res_'+str(x), filters, F, padding, True, False)

        if FPN:
            """
            FPN has two main differences to a Unet decoder:
            1. We don't decode all the way up to a feature map size equal to the original
            2. We don't decrease kernel sizes as we upsample
            """

            # Set first dconv to output of final conv after 1x1 conv. Also save intermediate outputs
            deconv = [None] * (self.nb_blocks-1)
            deconv[0] = self.convolution('FPNUp1', conv[-1], 1, FPN_layers, 1, phase_train=self.phase_train)

            # Now loop through and perform the upsamples. Don't go all the way to initial size
            for z in range(self.nb_blocks-2):

                # Z holds the prior index, X holds the layer index
                x = z + 1

                # First 1x1 conv the skip connection
                skip = self.convolution('FPNUp_'+str(x), conv[-(x+2)], 1, FPN_layers, 1, phase_train=self.phase_train)

                # Perform upsample unless at the end.
                if x < self.nb_blocks: deconv[x] = self.up_transition('Upsample_' + str(x), deconv[z], 3, FPN_layers, 2, skip, res=True)

                # Finally, perform the 3x3 conv without Relu
                deconv[x] = self.convolution('FPNOut_' + str(x), deconv[x], 3, FPN_layers, 1, phase_train=self.phase_train, BN=False, relu=False, bias=False)

            # Return the feature pyramid outputs
            return conv[-1], deconv

        # Return final layer and array of conv block outputs if no FPN
        else: return conv[-1], conv


class ResUNet(ResNet):

    """
    Creates UNet style residual networks
    """

    def __init__(self, nb_blocks, filters, sess, phase_train, summary):

        # Variables accessible to only specific instances here:

        self.nb_blocks = nb_blocks
        self.filters = filters
        self.sess = sess
        self.phase_train = phase_train
        self.summary = summary


    def dense_block(self, input_x, nb_layers, layer_name, keep_prob=None, downsample=False):

        """
        Creates a dense block
        :param input_x: The input to this dense block (output of prior downsample operation)
        :param nb_layers: how many layers desired
        :param layer_name: base name of this block
        :param keep_prob: Whether to use dropout
        :param trans: whether to include a downsample at the end
        :return:
        """

        with tf.name_scope(layer_name):

            # Array to hold each layer
            layers_concat = []

            # Append to list
            layers_concat.append(input_x)

            # The first layer of this block
            conv = self.bottleneck_layer(input_x, (layer_name + '_denseN_0'), keep_prob)

            # Concat the first layer
            layers_concat.append(conv)

            # Loop through the number of layers desired
            for z in range(nb_layers):
                # Concat all the prior layer into this layer
                conv = tf.concat(layers_concat, axis=-1)

                # Create a new layer
                conv = self.bottleneck_layer(conv, (layer_name + '_denseN_' + str(z + 1)), keep_prob)

                # Append this layer to the running list of dense connected layers
                layers_concat.append(conv)

            # Combine the layers
            conv = tf.concat(layers_concat, axis=-1)

            if downsample: conv = self.transition_layer(conv, (layer_name + '_Downsample'), keep_prob)

            return conv, layers_concat


    def dense_block_3d(self, input_x, nb_layers, layer_name, keep_prob=None):

        """
        Creates a dense block
        :param input_x: The input to this dense block (output of prior downsample operation)
        :param nb_layers: how many layers desired
        :param layer_name: base name of this block
        :param keep_prob: Whether to use dropout
        :return:
        """

        with tf.name_scope(layer_name):
            # Array to hold each layer
            layers_concat = []

            # Append to list
            layers_concat.append(input_x)

            # The first layer of this block
            conv = self.bottleneck_layer_3d(input_x, (layer_name + '_denseN_0'), keep_prob)

            # Concat the first layer
            layers_concat.append(conv)

            # Loop through the number of layers desired
            for z in range(nb_layers):
                # Concat all the prior layer into this layer
                conv = tf.concat(layers_concat, axis=-1)

                # Create a new layer
                conv = self.bottleneck_layer_3d(conv, (layer_name + '_denseN_' + str(z + 1)), keep_prob)

                # Append this layer to the running list of dense connected layers
                layers_concat.append(conv)

            # The concat has to be 2D, first retreive the Z and K size
            concat = tf.concat(layers_concat, -1)
            Fz, Ch = concat.get_shape().as_list()[1], concat.get_shape().as_list()[-1]

            # Now create a projection matrix
            concat = self.convolution_3d(layer_name + '_projection', concat, [Fz, 1, 1], Ch, 1, 'VALID', self.phase_train, BN=False)

            return conv, tf.squeeze(concat)


    def up_transition(self, scope, X, F, K, S, concat_var=None, padding='SAME'):

        """
        Deconvolutions for the DenseUnet
        :param scope:
        :param X:
        :param F:
        :param K:
        :param S:
        :param padding:
        :param concat_var:
        :param summary:
        :return:
        """

        with tf.variable_scope(scope) as scope:
            # Set channel size based on input depth
            C = X.get_shape().as_list()[-1]

            # He init
            kernel = tf.get_variable('Weights', shape=[F, F, K, C], initializer=tf.contrib.layers.variance_scaling_initializer())

            # Define the biases
            bias = tf.get_variable('Bias', shape=[K], initializer=tf.constant_initializer(0.0))

            # Add to the weights collection
            tf.add_to_collection('weights', kernel)
            tf.add_to_collection('biases', bias)

            # Define the output shape
            out_shape = X.get_shape().as_list()
            out_shape[1] *= 2
            out_shape[2] *= 2
            out_shape[3] = K

            # Perform the deconvolution. output_shape: A 1-D Tensor representing the output shape of the deconvolution op.
            conv = tf.nn.conv2d_transpose(X, kernel, output_shape=out_shape, strides=[1, S, S, 1], padding=padding)

            # Add in bias
            conv = tf.nn.bias_add(conv, bias)

            # Concatenate
            conv = tf.concat([concat_var, conv], axis=-1)

            # Create a histogram summary and summary of sparsity
            if self.summary: self._activation_summary(conv)

            return conv


    def up_transition_3d(self, scope, X, F, K, S, concat_var=None, padding='VALID'):

        """
        Deconvolutions for the DenseUnet
        :param scope:
        :param X:
        :param F:
        :param K:
        :param S:
        :param padding:
        :param concat_var:
        :param summary:
        :return:
        """

        with tf.variable_scope(scope) as scope:
            # Set channel size based on input depth
            C = X.get_shape().as_list()[-1]

            # He init
            kernel = tf.get_variable('Weights', shape=[F, F, F, K, C], initializer=tf.contrib.layers.variance_scaling_initializer())

            # Define the biases
            bias = tf.get_variable('Bias', shape=[K], initializer=tf.constant_initializer(0.0))

            # Add to the weights collection
            tf.add_to_collection('weights', kernel)
            tf.add_to_collection('biases', bias)

            # Define the output shape
            out_shape = X.get_shape().as_list()
            out_shape[1] *= 2
            out_shape[2] *= 2
            out_shape[3] *= 2
            out_shape[3] = K

            # Perform the deconvolution. output_shape: A 1-D Tensor representing the output shape of the deconvolution op.
            conv = tf.nn.conv2d_transpose(X, kernel, output_shape=out_shape, strides=[1, S, S, S, 1], padding=padding)

            # Add in bias
            conv = tf.nn.bias_add(conv, bias)

            # Concatenate
            conv = tf.concat([concat_var, conv], axis=-1)

            # Create a histogram summary and summary of sparsity
            if self.summary: self._activation_summary(conv)

            return conv


    def define_network(self, layers=[], keep_prob=None):

        # conv holds output of bottleneck layers (no BN/ReLU). Concat holds running lists of skip connections
        conv, concat = [None] * (self.nb_blocks + 1), [None] * (self.nb_blocks - 1)

        # Define the first layers before starting the Dense blocks
        conv[0] = self.convolution('Conv1', self.images, 3, 2 * self.filters, 1, phase_train=self.phase_train, BN=False, relu=False)

        # Loop through and make the downsample blocks
        for z in range(self.nb_blocks):

            # Z holds the prior index, X holds the layer index
            x = z + 1

            # Generate a dense block
            conv[x], concat[z] = self.dense_block(conv[z], layers[z], 'Dense_' + str(x), keep_prob)

            # Downsample unless at the end
            if x < self.nb_blocks: conv[x] = self.transition_layer(conv[x], 'Downsample_' + str(x), keep_prob)

        # Set first dconv to output of final conv
        deconv = tf.nn.relu(self.batch_normalization(conv[-1], self.phase_train, None))

        # Now loop through and perform the upsamples. No longer need to store these
        for z in range(self.nb_blocks):

            # Z holds the prior index, X holds the layer index
            x = z + 1

            # Perform upsample unless at the end
            if x < self.nb_blocks: deconv = self.up_transition('Upsample_' + str(x), deconv, 3, self.filters, 2, concat[-x])

            # Generate a dense block
            deconv, _ = self.dense_block(deconv, layers[-(z + 2)], 'Up_Dense_' + str(x), keep_prob)

        # Return final layer after batch norm and relu
        return tf.nn.relu(self.batch_normalization(deconv, self.phase_train, 'BNa'))


    def define_network_25D(self, layers=[], keep_prob=None, prints=True):

        # conv holds output of bottleneck layers (no BN/ReLU). Concat holds running lists of skip connections
        conv, concat = [None] * (self.nb_blocks + 1), [None] * (self.nb_blocks - 1)

        # Define the first layers before starting the Dense blocks
        conv[0] = self.convolution_3d('Conv1', self.images, [1, 3, 3], 2 * self.filters, 1, 'SAME', self.phase_train, BN=False, relu=False)

        # Loop through and make the downsample blocks
        for z in range(self.nb_blocks):

            # Z holds the prior index, X holds the layer index
            x = z + 1

            # Retreive number of slices here
            slices = conv[z].get_shape().as_list()[1]

            # Error handling for when we are down to rank 4
            if len(conv[z].get_shape().as_list()) == 4: slices = 1

            # Generate a dense block. 3D if slices > 1
            if slices > 1:
                conv[x], concat[z] = self.dense_block_3d(conv[z], layers[z], 'Dense_' + str(x), keep_prob)
            else:
                conv[x], _ = self.dense_block(tf.squeeze(conv[z]), layers[z], 'Dense_' + str(x), keep_prob)

            # Downsample unless at the end
            if x < self.nb_blocks and slices > 1:
                conv[x] = self.transition_layer_3d(conv[x], 'Downsample_' + str(x))
            elif x < self.nb_blocks and slices == 1:
                conv[x] = self.transition_layer(conv[x], 'Downsample_' + str(x))

        # Set first dconv to output of final conv
        deconv = tf.nn.relu(self.batch_normalization(conv[-1], self.phase_train, None))

        # Now loop through and perform the upsamples. No longer need to store these. These are 2D
        for z in range(self.nb_blocks - 1):

            # Z holds the prior index, X holds the layer index
            x = z + 1

            # Perform upsample unless at the end
            if x < self.nb_blocks: deconv = self.up_transition('Upsample_' + str(x), deconv, 3, self.filters, 2, concat[-x])

            # Generate a dense block
            deconv, _ = self.dense_block(deconv, layers[-(z + 2)], 'Up_Dense_' + str(x), keep_prob)

        # Return final layer after batch norm and relu
        return tf.nn.relu(self.batch_normalization(deconv, self.phase_train, None))


class MRCNN(SODMatrix):

    """
    The multiple inheritence class to perform all functions of making mask RCNNs!
    """

    # Shared class variables here

    def __init__(self, phase_train, GPU_count=1, Images_per_gpu=2, FCN=256, K1=32, Num_classes=1, FPN_layers=64,
                 RPN_anchor_scales= (1, 0.5), RPN_anchor_ratios = [0.5, 1, 2], Image_size=512, RPN_base_anchor_size = [6, 13, 20, 35],
                 RPN_nms_upper_threshold = 0.7, RPN_nms_lower_threshold=0.3, RPN_anchors_per_image=256,
                 POST_NMS_ROIS_training=2000, POST_NMS_ROIS_testing=1000, Use_mini_mask=False, Mini_mask_shape = [56, 56]):

        """
        Instance specific variables
        :param phase_train: Training or testing mode
        :param GPU_count: Numbger of GPUs to use
        :param Images_per_gpu: Number of images to train with on each GPU. Use highest number gpu can handle
        :param FCN: Number of neurons in the first fully connected layer
        :param K1: Number of filters in the first layer
        :param Num_classes: Number of classification classes
        :param RPN_anchor_scales: Length of square anchor side in pixels
        :param RPN_anchor_ratios: Ratios of anchors at each cell (width/height) 1 represents a square anchor, and 0.5 is a wide anchor
        :param: Image_size: The size of the input images
        :param: base_anchor_size: The list of anchor sizes for [32, 64, 128, 256] feature map sizes
        :param RPN_nms_upper_threshold: Non-max suppression threshold to filter RPN proposals. Increase for more proposals
        :param RPN_nms_lower_threshold: Non-min suppression threshold to filter RPN proposals. Decrease for more proposals
        :param RPN_anchors_per_image: ow many anchors per image to use for RPN training
        :param POST_NMS_ROIS_training: ROIs kept after non-maximum supression (training)
        :param POST_NMS_ROIS_testing: ROIs kept after non-maximum supression (inference)
        :param Use_mini_mask: If enabled, resizes instance masks to a smaller size to reduce memory usage
        :param Mini_mask_shape: (height, width) of the mini-mask
        """

        self.phase_train = phase_train
        self.GPU_count = GPU_count
        self.Images_per_gpu = Images_per_gpu
        self.FCN = FCN
        self.K1 = K1
        self.Num_classes = Num_classes
        self.FPN_layers = FPN_layers
        self.RPN_anchor_scales = RPN_anchor_scales
        self.RPN_anchor_ratios = RPN_anchor_ratios
        self.RPN_nms_upper_threshold = RPN_nms_upper_threshold
        self.RPN_nms_lower_threshold = RPN_nms_lower_threshold
        self.RPN_anchors_per_image = RPN_anchors_per_image
        self.POST_NMS_ROIS_training = POST_NMS_ROIS_training
        self.POST_NMS_ROIS_testing = POST_NMS_ROIS_testing
        self.Use_mini_mask = Use_mini_mask
        self.Mini_mask_shape = Mini_mask_shape
        self.Image_size = Image_size
        self.RPN_base_anchor_size  = RPN_base_anchor_size

        # Keeping track of the layers and losses
        self.RPN_conv = None
        self.RPN_ROI = None
        self.RPN_Loss_Object = None
        self.RPN_Loss_Box = None
        self.RPN_class_logits = None
        self.RPN_bbox_score = None


    """
    Baseline Networks
    """

    # Overwrite the convolution function wrapper to include the reuse flag for our RPN to work on a FPN output. Remove downsample and dropout
    def convolution_RPN(self, scope, X, F, K, S=1, padding='SAME', phase_train=None, BN=True, relu=True, bias=True, reuse=None, summary=True):
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


    def RCNN_base(self, input_images, net_type, input_dims, FPN=True):

        """
        Builds the base feature pyramid network
        :param input_images: the input images
        :param net_type: can be a ResNet, DenseNet or Inception net
        :param input_dims: dimensions of the input images
        :param FPN: Whether to use a feature pyramid network (you should)
        :return:
                The final feature map outputs from the feature pyramid network
        """

        # Calculate how many blocks are needed to get to a final dimension of 32x32
        nb_blocks = int((input_dims/32) ** (0.5)) + 1

        # Calculate block layers downsampling scheme
        block_sizes = [None] * nb_blocks
        for z in range(nb_blocks): block_sizes[z] = 2 + 2*z

        if net_type == 'RESIDUAL':

            # Define a ResNet
            self.resnet = ResNet(nb_blocks=nb_blocks, filters=self.K1, images=input_images, sess=None, phase_train=self.phase_train, summary=True)

            # Make sure all the 64x64 and below feature maps are inception style
            inception_layers = [None] * nb_blocks
            inception_layers[-1], inception_layers[-2] = 1, 1

            # Define the downsample network and retreive the output of each block #TODO: Fix class inheritence and FPN layers
            if FPN: final_conv, self.conv = self.resnet.define_network(block_sizes, inception_layers, FPN=True, FPN_layers=self.FPN_layers)
            else: final_conv, self.conv = self.resnet.define_network(block_sizes, inception_layers, FPN=False)

            return final_conv, self.conv

        if net_type == 'DENSE':
            pass


    def RPN_predict(self):

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
            reuse_flag = None if lvl==0 else True
            scope_list = ['RPN_3x3', 'RPN_Classifier', 'RPN_Regressor']

            # Now run a 3x3 conv, then separate 1x1 convs for each feature map #TODO: Check if VALID, check BN/Relu
            rpn_3x3 = self.convolution_RPN(scope_list[0], self.conv[lvl], 3, self.FPN_layers, phase_train=self.phase_train, reuse=reuse_flag, padding='VALID')
            rpn_class = self.convolution_RPN(scope_list[1], rpn_3x3, 1, num_class_scores, BN=False, relu=False, bias=False, phase_train=self.phase_train, reuse=reuse_flag, padding='VALID')
            rpn_box = self.convolution_RPN(scope_list[2], rpn_3x3, 1, num_bbox_scores, BN=False, relu=False, bias=False, phase_train=self.phase_train, reuse=reuse_flag, padding='VALID')

            # Reshape the scores and append to the list
            rpn_class, rpn_box = tf.reshape(rpn_class, [-1, 2]), tf.reshape(rpn_box, [-1, 4])
            class_logits.append(rpn_class)
            box_logits.append(rpn_box)

        # Return the concatenated list
        return tf.concat(class_logits, axis=0), tf.concat(box_logits, axis=0)


    def RCNN_RPN(self):

        """
        Makes anchors and runs the RPN forward pass
        :return:
        """

        # Run the forward pass and generate anchor boxes
        class_logits, box_logits = self.RPN_predict()
        anchors = self.make_anchors_FPN()

        # Under new name scope (not get_variable), clean up the anchors generated during training
        with tf.name_scope('RPNN_Forward'):
            if self.training_phase:

                # Remove outside anchor boxes by returning only the indices of the valid anchors
                valid_indices = self.filter_outside_anchors(anchors, self.Image_size)
                valid_anchors = tf.gather(anchors, valid_indices)
                valid_cls_logits, valid_box_logits = tf.gather(class_logits, valid_indices), tf.gather(box_logits, valid_indices)

                # Return the valid anchors, else during trainign just return the anchors
                return valid_anchors, valid_cls_logits, valid_box_logits

            else: return anchors, class_logits, box_logits


    """
    Region proposal box generating functions, aka "anchor" functions:
    Make anchors calls generate_anchors for each feature map level in the RPN
    """

    def make_anchors_FPN(self):

        # Var scope creates a unique scope for all variables made including with tf.get_variable. name_scope allows reusing of variables
        with tf.variable_scope('make_anchors'):

            # List of anchors
            anchor_list, level_list = [], self.conv

            # Name scope doesn't rescope tf.get_variable calls
            with tf.name_scope('make_anchors_all_levels'):

                # Loop through each FPN feature map scale
                for FPN_scale, base_anchor_size in zip(self.conv, self.RPN_base_anchor_size):

                    # Stride = usually 1, return feature map size. Use index 2 in case of 3D feature maps
                    fm_size = tf.cast(tf.shape(FPN_scale)[2], tf.float32)

                    # Calculate feature stride
                    feature_stride = tf.cast(self.Image_size // fm_size, tf.float32)

                    # Base anchor sizes match feature map sizes. Generate anchors at this level
                    anchors = self.generate_anchors(fm_size, base_anchor_size, feature_stride, self.RPN_anchor_ratios, self.RPN_anchor_scales)

                    # Reshape and append to the list
                    anchors = tf.reshape(anchors, [-1, 4])
                    anchor_list.append(anchors)

                all_level_anchors = tf.concat(anchor_list, axis=0)

            return all_level_anchors


    def generate_anchors(self, shape, base_anchor_size, feature_stride, ratios, scales, name='generate_anchors'):

        """
        For generating anchors inside the tensorflow computation graph
        :param shape: Spatial shape of the feature map over which to generate anchors
        :param base_anchor_size: The base anchor size for this feature map
        :param ratios: [1D array] of anchor ratios of width/height. i.e [0.5, 1, 2]
        :param scales: [1D array] of anchor scales in original space
        :param feature_stride: int, stride of feature map relative to the image in pixels
        :param anchor_stride: int, stride of anchors on the feature map
        :return:
        """

        # Define a variable scope
        with tf.variable_scope(name):

            # Generate a base anchor
            base_anchor = tf.constant([0, 0, base_anchor_size, base_anchor_size], tf.float32)
            base_anchors = self.enum_ratios(self.enum_scales(base_anchor, scales), ratios)
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



    def filter_outside_anchors(self, anchors, img_dim):

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



    """
    Bounding box functions
    """

    def extract_box_labels(self, mask, dim_3d=False):

        """
        Returns the bounding box labels from each 2D segmentation mask. Can work with multiple label groupings (classes)
        :param mask: Input mask in 2D [height, weight, channels] or 3D [slice, height, weight, channels]
        :param dim_3d: bool, whether input is 3D or 2D
        :return: np list of arrays of bounding box coordinates of the corners [N, (y1, x1, y2, x2)]
        """

        if dim_3d:

            # Expand dims
            if mask.ndim < 4: mask = np.expand_dims(mask, axis=-1)

            # Make dummy array with diff boxes for each channel
            boxes = np.zeros([mask.shape[0], mask.shape[-1], 4], dtype=np.int32)

            for a in range (mask.shape[0]):

                # Loop through all the classes (channels)
                for z in range(mask[a].shape[-1]):

                    # Work on just this class of pixel
                    m = mask[a, :, :, z]

                    # Bounding boxes generated by finding indices with true values
                    x_indices = np.where(np.any(m, axis=0))[0]
                    y_indices = np.where(np.any(m, axis=1))[0]

                    if x_indices.shape[0]:

                        x1, x2 = x_indices[[0, -1]]
                        y1, y2 = y_indices[[0, -1]]

                        # Increment x2 and y2 by 1 since theyre not part of the initial box
                        x2 += 1
                        y2 += 1

                    else:

                        # No mask for this instance, happens a lot. Sets boxes to zero
                        x1, x2, y1, y2 = 0, 0, 0, 0

                    # Append coordinates to boxes
                    boxes[a, z] = np.array([y1, x1, y2, x2])

            return boxes.astype(np.int32)

        else:

            # Expand dims
            if mask.ndim < 3: mask = np.expand_dims(mask, axis=-1)

            # Make dummy array with diff boxes for each channel
            boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)

            # Loop through all the classes (channels)
            for z in range (mask.shape[-1]):

                # Work on just this class of pixel
                m = mask[:, :, z]

                # Bounding boxes generated by finding indices with true values
                x_indices = np.where(np.any(m, axis=0))[0]
                y_indices = np.where(np.any(m, axis=1))[0]

                if x_indices.shape[0]:

                    x1, x2 = x_indices[[0, -1]]
                    y1, y2 = y_indices[[0, -1]]

                    # Increment x2 and y2 by 1 since theyre not part of the initial box
                    x2 += 1
                    y2 += 1

                else:

                    # No mask for this instance, happens a lot. Sets boxes to zero
                    x1, x2, y1, y2 = 0, 0, 0, 0

                # Append coordinates to boxes
                boxes[z] = np.array([y1, x1, y2, x2])

            return boxes.astype(np.int32)


    def calculate_iou(self, box, boxes, box_area, boxes_area):

        """
        Function that calculates intersection over union of the given box with the given array of boxes
        :param box: The given box corner coordinates [y1, x1, y2, x2]
        :param boxes: Boxes to check [N, (y1, x1, y2, x2)]
        :param box_area: float, the area of 'box'
        :param boxes_area: array of length boxes_count
        :return: intersection over union
        """

        # retreive box coordinates
        y1, y2 = np.maximum(box[0], boxes[:, 0]), np.minimum(box[2], boxes[:, 2])
        x1, x2 = np.maximum(box[1], boxes[:, 1]), np.minimum(box[3], boxes[:, 3])

        # Calculate the intersection over union
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        union = box_area + boxes_area[:] - intersection[:]

        return intersection / union


    def calculate_overlaps_IoU_boxes(self, boxes1, boxes2):

        """
        Calculates the IoU overlaps between two sets of boxes. Pass smaller set second for best performance
        :param boxes1: [N, (y1, x1, y2, x2)].
        :param boxes2: [N, (y1, x1, y2, x2)].
        :return: overlaps
        """

        # Area of anchors and ground truth boxes
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        # Compute overlaps to generate matrix [boxes1 count, boxes2 count] where each cell contains the IoU value.
        overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
        for i in range(overlaps.shape[1]):
            box2 = boxes2[i]
            overlaps[:, i] = self.calculate_iou(box2, boxes1, area2[i], area1)

        return overlaps


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


    def generate_deltas(self, anchors, offsets):

        """
        Function that generates the deltas from the RPN
        :param anchors:
        :param offsets:
        :return:
        """

        deltas = np.zeros(offsets.shape)
        idx = np.sum(np.abs(anchors), axis=1) > 0
        idx = np.nonzero(idx)[0]

        for z in idx:

            # Retreive the source and target coordinates
            y, x, h, w = offsets[z]
            y1, x1, y2, x2 = anchors[z]

            # Retreive center point and mean transform
            ya, xa = np.mean((y1, y2)), np.mean((x1, x2))
            h2, w2 = (y2 - y1), (x2 - x1)

            # Generate the delta array. Avoid numerical errors
            deltas[z] = [(ya - y) / h, (xa - x) / w, np.log(h2 / (h+1e-6)), np.log(w2 / (w + 1e-6))]

            return deltas.astype('float32')


    def calculate_box_deltas(self, box, label_box):

        """
        Calculates the refinement needed to transform predicted box to the label (ground truth) box
        :param box: [N, (y1, x1, y2, x2)]
        :param label_box: [N, (y1, x1, y2, x2)]
        :return:
        """

        # Convert to float32
        box, label_box = tf.cast(box, tf.float32), tf.cast(label_box, tf.float32)

        # Convert to y, x, h, w
        h, w = (box[:, 2] - box[:, 0]), (box[:, 3] - box[:, 1])
        center_y, center_x = (h * 0.5 + box[:, 0]), (w * 0.5 + box[:, 1])
        lbl_h, lbl_w = (label_box[:, 2] - label_box[:, 0]), (label_box[:, 3] - label_box[:, 1])
        lbl_center_y, lbl_center_x = (lbl_h * 0.5 + label_box[:, 0]), (lbl_w * 0.5 + label_box[:, 1])

        # Now calculate the normalized shifts
        dy = (lbl_center_y - center_y) / h
        dx = (lbl_center_x - center_x) / w
        dh, dw = tf.log(lbl_h / h), tf.log(lbl_w / w)

        return tf.stack([dy, dx, dh, dw], axis=1)


    def apply_deltas(self, boxes, deltas):

        """
        Applies the deltas given to the boxes given
        :param boxes: [N, (y1, x1, y2, x2)]
        :param deltas: [N, (dy, dx, log(dh), log(dw)]
        :return:
        """

        # Fix data type
        boxes = boxes.astype(np.float32)

        # Convert to y, x, h, w
        h, w = (boxes[:, 2] - boxes[:, 0]), (boxes[:, 3] - boxes[:, 1])
        center_y, center_x = (h * 0.5 + boxes[:, 0]), (w * 0.5 + boxes[:, 1])

        # Apply deltas
        center_y += deltas[:, 0] * h
        center_x += deltas[:, 1] * w

        # Convert back to y1, x1, y2, x2
        y1, x1 = (center_y - 0.5*h), (center_x - 0.5*w)
        y2, x2 = (y1 + h), (x1 + w)

        return np.stack([y1, x1, y2, x2], axis=1)


    """
    Testing functions
    """

    def trim_zeros(self, x):

        """
        Remove all rows of a tensor that are all zeros
        :param x: [rows, columns]
        :return:
        """

        assert len(x.shape == 2)
        return x[~np.all(x==0, axis=1)]


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
                    match_count +=1
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
            if verbose: print ('PPV: %.2f: \t %.3f' %(iou_threshold, ppv))

        PPV = np.array(PPV).mean()
        if verbose: print ('PPV: %.2f: \t %.3f: \t %.3f' %(iou_thresholds[0], iou_thresholds[-1], PPV))

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
    Utility Functions
    """

    def batch_slice(self, inputs, graph_fn, batch_size, names=None):

        """
        Splits inputs into slices and feeds each slice to a copy of the given computation graph and then combines the results.
        It allows you to run a graph on a batch of inputs even if the graph is written to support one instance only.
        :param inputs: List of tensors with same first dimension length
        :param graph_fn: function that returns a TF tensor thats part of a graph
        :param batch_size: Number of slices to divide the data into
        :param names: Assigns names to the resulting tensors if provided
        :return:
        """

        # Change the inputs to a list
        if not isinstance(inputs, list): inputs = [inputs]

        outputs = []
        for i in range(batch_size):
            inputs_slice = [x[i] for x in inputs]
            output_slice = graph_fn(*inputs_slice)
            if not isinstance(output_slice, (tuple, list)): output_slice = [output_slice]
            outputs.append(output_slice)

        # Change outputs from a list of slices where each is a list of outputs to a list of outputs and each has a list of slices
        outputs = list(zip(*outputs))

        if names is None: names = [None] * len(outputs)

        result = [tf.stack(o, axis=0, name=n) for o, n in zip(outputs, names)]
        if len(result) == 1: result = result[0]

        return result


    def norm_boxes(self, boxes, shape):

        """
        Normalizes the pixel coordinates of boxes
        :param boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
        :param shape: [..., (height, width)] in pixels
                Note: In pixel coordinates (y2, x2) is outside the box. But in normalized coordinates it's inside the box.
        :return: [N, (y1, x1, y2, x2)] in normalized coordinates
        """

        h, w = shape
        scale = np.array([h - 1, w - 1, h - 1, w - 1])
        shift = np.array([0, 0, 1, 1])
        return np.divide((boxes - shift), scale).astype(np.float32)


    def denorm_boxes(self, boxes, shape):

        """
        Converts boxes from normalized coordinates to pixel coordinates.
        :param boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
        :param shape: [..., (height, width)] in pixels
                Note: In pixel coordinates (y2, x2) is outside the box. But in normalized coordinates it's inside the box.
        :return: [N, (y1, x1, y2, x2)] in pixel coordinates
        """

        h, w = shape
        scale = np.array([h - 1, w - 1, h - 1, w - 1])
        shift = np.array([0, 0, 1, 1])
        return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


    def enum_scales(self, base_anchor, anchor_scales, name='enum_scales'):

        '''
        :param base_anchor: [y_center, x_center, h, w]
        :param anchor_scales: different scales, like [0.5, 1., 2.0]
        :return: return base anchors in different scales.
                Example:[[0, 0, 128, 128],[0, 0, 256, 256],[0, 0, 512, 512]]
        '''
        with tf.variable_scope(name):

            anchor_scales = base_anchor * tf.constant(anchor_scales, dtype=tf.float32, shape=(len(anchor_scales), 1))
            return anchor_scales

    def enum_ratios(self, anchors, anchor_ratios, name='enum_ratios'):

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

    # def enum_ratios(self, anchors, anchor_ratios):
    #
    #     '''
    #     ratio = h /w
    #     :param anchors:
    #     :param anchor_ratios:
    #     :return:
    #     '''
    #     ws = anchors[:, 2]  # for base anchor: w == h
    #     hs = anchors[:, 3]
    #     sqrt_ratios = tf.sqrt(tf.constant(anchor_ratios))
    #
    #     ws = tf.reshape(ws / sqrt_ratios[:, tf.newaxis], [-1, 1])
    #     hs = tf.reshape(hs * sqrt_ratios[:, tf.newaxis], [-1, 1])
    #
    #     return hs, ws
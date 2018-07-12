"""
SOD_DenseNet contains the class utilized in implementing a residual network
"""

from SODNetwork import tf
from SODNetwork import np
from SODNetwork import SODMatrix


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
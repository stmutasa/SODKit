"""
SOD_ResNet contains the class utilized in implementing a residual network
"""

from SODNetwork import tf
from SODNetwork import np
from SODNetwork import SODMatrix


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


    def residual_block_3d(self, input_x, nb_layers, layer_name, K, F=3, padding='SAME', downsample=2, stanford=False):

        """
        Implements a block of residual layers at the same spatial dimension in 3 dimensions
        :param input_x: Input, either from the last conv layer or the images
        :param nb_layers: number of residual layers
        :param layer_name: the baseline name of this block
        :param K: feature map size
        :param F: filter size
        :param padding: SAME or VALID
        :param downsample: 0 = None, 1 = Traditional, 2 = 2.5D downsample
        :param stanford: whether to use stanford style layers
        :return:
        """

        with tf.name_scope(layer_name):

            # The first layer of this block
            conv = self.residual_layer_3d((layer_name + '_res_0'), input_x, F, K, 1, padding, self.phase_train)

            # Loop through the number of layers desired
            for z in range(nb_layers):

                # Perform the desired operations
                conv = self.residual_layer_3d((layer_name + '_res_' + str(z + 1)), conv, F, K, 1, padding, self.phase_train)

            # Perform a downsample. 0 = None, 1 = Traditional, 2 = 2.5D downsample
            if downsample == 1: conv = self.convolution_3d((layer_name + '_res_down_'), conv, F, K*2, 2, padding, self.phase_train)

            elif downsample == 2: conv = self.convolution_3d((layer_name + '_res_down_'), conv, [2, 3, 3], K*2, [1, 2, 2], 'VALID', self.phase_train)

            return conv


    def inception_block_3d(self, input_x, nb_layers, layer_name, K, padding='SAME', downsample=2):

        """
        Implements a block of inception layers at the same spatial dimension in 3 dimensions
        :param input_x: Input, either from the last conv layer or the images
        :param nb_layers: number of layers
        :param layer_name: the baseline name of this block
        :param F: filter size
        :param padding: SAME or VALID
        :param downsample: 0 = None, 1 = Traditional, 2 = 2.5D downsample
        :return:
        """

        with tf.name_scope(layer_name):

            # The first layer of this block
            conv = self.inception_layer_3d((layer_name + '_inc_0'), input_x, K, 1, 1, padding, self.phase_train)

            # Loop through the number of layers desired
            for z in range(nb_layers):

                # Perform the desired operations
                conv = self.inception_layer_3d((layer_name + '_inc_' + str(z + 1)), conv, K, 1, 1, padding, self.phase_train)

            # Perform a downsample. 0 = None, 1 = Traditional, 2 = 2.5D downsample
            if downsample == 1: conv = self.convolution_3d((layer_name + '_res_down_'), conv, 3, K * 2, 2, padding, self.phase_train)

            elif downsample == 2: conv = self.convolution_3d((layer_name + '_res_down_'), conv, [2, 3, 3], K * 2, [1, 2, 2], 'VALID', self.phase_train)

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
            if downsample: conv = self.inception_layer((layer_name + '_inc_down_'), conv, K * 2, 2, padding, self.phase_train)

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
        :param padding: SAME or VALID. In general, use VALID for 3D skip connections
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

            # Define the output shape based on shape of skip connection
            out_shape = concat_var.get_shape().as_list()
            out_shape[-1] = K

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


    def define_network_25D(self, block_layers=[], inception_layers=[], F=3, S_1=1, padding='SAME', downsample_last=False, FPN=True, FPN_layers=64):

        """
        Shortcut to creating a residual or residual-inception style network with just a few lines of code
        Additionally, we can create a feature pyramid network at the end
        Please note, for 2.5 D implementations, the first Conv maintains the same feature map size so make sure to not have too many blocks here!
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
        conv[0] = self.convolution_3d('Conv1', self.images, [2, F, F], self.filters, S_1, phase_train=self.phase_train)

        # To save filters for later
        filter_size_buffer = []

        # Loop through and make the downsample blocks
        for z in range (self.nb_blocks):

            # Z holds the prior index, X holds the current layer index
            x = z+1

            # Retreive number of z dimensions here
            z_dim = conv[z].get_shape().as_list()[1]

            # When down to rank 4, we must be at 2 dimensions [batch, z, y, x, c] --> [batch, y, x, c]
            if len(conv[z].get_shape().as_list()) == 4: z_dim = 1

            # Set filter size for this block
            if S_1 == 1: filters = self.filters * (2**z)
            else: filters = self.filters * (2**x)
            filter_size_buffer.append(filters)

            if z_dim > 1:

                # Generate the appropriate 3D only downsample if not at the end
                if inception_layers[z]:
                    if (not downsample_last) and x == self.nb_blocks: conv[x] = self.inception_block_3d(conv[z], block_layers[z], 'Inc_' + str(x), filters, padding, 0)
                    else: conv[x] = self.inception_block_3d(conv[z], block_layers[z], 'Inc_' + str(x), filters, padding, 2)

                else:
                    if (not downsample_last) and x == self.nb_blocks: conv[x] = self.residual_block_3d(conv[z], block_layers[z], 'Res_' + str(x), filters, F, padding, 0, False)
                    else:  conv[x] = self.residual_block_3d(conv[z], block_layers[z], 'Res_' + str(x), filters, F, padding, 2, False)

            else:

                # Now for the 2D blocks
                conv[z] = tf.squeeze(conv[z])
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

                # Retreive the shape of the skip connection
                skip_z = conv[-(x + 2)].get_shape().as_list()[1]

                # If this is 2D, perform SAME convolutions, else perform VALID
                if len(conv[z].get_shape().as_list()) == 4: skip_z, padding = 1, 'SAME'
                else: padding = 'VALID'

                # First 1x1 conv the skip connection
                if skip_z==1: skip = self.convolution('FPNUp_'+str(x), conv[-(x+2)], 1, FPN_layers, 1, phase_train=self.phase_train)
                else: skip = self.convolution_3d('Projection_' + str(x), conv[-(x+2)], [skip_z, 1, 1], FPN_layers, 1, 'VALID', self.phase_train, BN=False, relu=False)

                # Perform upsample unless at the end.
                if x < self.nb_blocks: deconv[x] = self.up_transition('Upsample_' + str(x), deconv[z], 3, FPN_layers, 2, tf.squeeze(skip), padding=padding, res=True)

                # Finally, perform the 3x3 conv without Relu
                deconv[x] = self.convolution('FPNOut_' + str(x), deconv[x], 3, FPN_layers, 1, phase_train=self.phase_train, BN=False, relu=False, bias=False)

            # Return the feature pyramid outputs
            return conv[-1], deconv

        # Return final layer and array of conv block outputs if no FPN
        else: return conv[-1], conv


class ResUNet(ResNet):

    """
    Creates UNet style residual networks
    TODO: Work in progress to convert from DenseNet version
    Must get rid of transition layer and bottleneck layer
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


    def define_network_25D(self, layers=[], keep_prob=None):

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


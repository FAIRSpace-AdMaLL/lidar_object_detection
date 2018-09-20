'''
Tempo-Spatial LSTM model implementation using Tensorflow
Social LSTM Paper: http://vision.stanford.edu/pdf/CVPR16_N_LSTM.pdf

Author : Anirudh Vemula
Date: 10th October 2016
'''

import tensorflow as tf
import layers
import numpy as np

# The MLP model
class Model():

    def __init__(self, args, is_training=True):
        '''
        Initialisation function for the class Model.
        Params:
        args: Contains arguments required for the Model creation
        '''

        # Store the arguments
        self.args = args

        # Initialize a MLP
        if args.initializer == 'svd':
            initializer = tf.orthogonal_initializer()
        elif args.initializer == "xavier":
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = None

        # Input data contains sequence of (x,y) points
        self.input_data = tf.placeholder(tf.float32, [None, args.input_dim[0], args.input_dim[1], args.input_dim[2]])
        # target data contains sequences of (x,y) points as well
        self.target_data = tf.placeholder(tf.uint8, [None])

        # Learning rate
        self.lr = tf.Variable(args.learning_rate, trainable=False, name="learning_rate")
        #self.is_training = tf.Variable(is_training, trainable=False, name="is_training")
        #self.step = tf.Variable(0.0, trainable=False, name="step")

        output_dim = args.output_dim

        # MLP
        net = self.input_data
        for i, (mpl_size_i, kernel_size_i) in enumerate(zip(args.mpl_size, args.kernel_size)):
            net = tf.layers.conv2d(net, mpl_size_i, kernel_size_i, padding='valid', data_format='channels_last')
            net = tf.nn.relu(net)
            # net = tf.layers.batch_normalization(net, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)


        # Global Pooling
        net = tf.layers.max_pooling2d(net, [args.input_dim[0], 1], strides=[1, 1], padding='VALID')
        net = tf.layers.flatten(net)
        # code = tf.nn.l2_normalize(code, dim=1)
        # Output linear layer

        for i, mpl in enumerate(args.classification_layers):
            net = tf.layers.dense(net, mpl)
            net = tf.nn.relu(net)
            # net = tf.layers.batch_normalization(net, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)

            # Dropout
            if not is_training:
                args.keep_pro = 1.0

            net = tf.nn.dropout(net, keep_prob=args.keep_prob)

        # visual feature before classification
        self.feature = net
        net = tf.layers.dense(net, output_dim)

        self.output = tf.nn.softmax(net)
        self.pred = tf.argmax(self.output, axis=1)

        if is_training:
            label = tf.one_hot(self.target_data, args.output_dim)
            loss = tf.losses.softmax_cross_entropy(label, net)
            self.loss = loss

            # Get trainable_variables
            tvars = tf.trainable_variables()

            '''
            # L2 loss
            #l2 = args.lambda_param * sum(tf.nn.l2_loss(tvar) for tvar in tvars)
            #self.cost = self.cost + l2

            # Sparsenss loss
            #log_code = tf.log(code)
            #sparse_loss = tf.reduce_sum(code, axis=1)
            #sparse_loss = tf.exp(sparse_loss)
            #loss += 0.000001 * tf.reduce_sum(sparse_loss)
            '''

            #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            #with tf.control_dependencies(update_ops):
            self.gradients = tf.gradients(loss, tvars)
            # Clip the gradients if they are larger than the value given in args
            grads, _ = tf.clip_by_global_norm(self.gradients, args.grad_clip)

            #optimizer = tf.train.AdamOptimizer(self.lr)
            # initialize the optimizer with teh given learning rate
            # optimizer = tf.train.RMSPropOptimizer(self.lr)
            optimizer = tf.train.AdamOptimizer(self.lr)
            # optimizer = tf.train.GradientDescentOptimizer(self.lr)
            # optimizer = tf.train.AdadeltaOptimizer(self.lr)

            # Train operator
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))


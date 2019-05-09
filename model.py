from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import sys

# 95% accuracy
# 16 channels
# 5x5 kernel conv
# 4x4 kernel max_pool
# 16 fc1
# nbClasses fc2
def tiny_model(input, nbClasses):
        # Input sizes (to be extracted automatically)
        input_size = 32
        input_channels = 3
        # NN structure parameters
        kernels = [7,3]
        poolings = [4,4]
        conv_features = [8,32]
        fc_layer_output = 32
        # Count of parameters
        manual_nb_neurons = {}
        with tf.name_scope("tiny_model"):
                with tf.variable_scope("conv1") as scope:
                        net = tf.contrib.layers.conv2d(input, conv_features[0], [kernels[0], kernels[0]],
                                                       activation_fn=tf.nn.relu, padding='SAME',
                                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                       scope=scope)
                        # net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope=scope)
                        net = tf.contrib.layers.max_pool2d(net, [poolings[0], poolings[0]], padding='VALID')
                        manual_nb_neurons["conv1"] = input_channels * conv_features[0] * kernels[0] ** 2
                        
                with tf.variable_scope("conv2") as scope:
                        net = tf.contrib.layers.conv2d(net, conv_features[1], [kernels[1], kernels[1]],
                                                       activation_fn=tf.nn.relu, padding='SAME',
                                                       weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                       scope=scope)
                        # net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope=scope)
                        net = tf.contrib.layers.max_pool2d(net, [poolings[1], poolings[1]], padding='VALID')
                        manual_nb_neurons["conv2"] = conv_features[0] * conv_features[1] * kernels[1] ** 2
                        
                # for opencv3.2
                net_shape = net.get_shape().as_list()
                net = tf.reshape(net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
                
                # for opencv4.1 and opencv3.4.6
                # net = tf.contrib.layers.flatten(net)
                
                with tf.variable_scope("fc1") as scope:
                        net = tf.contrib.layers.fully_connected(net, fc_layer_output, activation_fn=tf.nn.relu,
                                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                                scope=scope)
                        layer_input_dim = conv_features[1] * (input_size / poolings[0] / poolings[1]) ** 2
                        manual_nb_neurons["fc1"] = layer_input_dim * fc_layer_output

                with tf.variable_scope("output") as scope:
                        net = tf.contrib.layers.fully_connected(net, nbClasses, activation_fn=tf.nn.softmax,
                                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                                scope=scope)
                        manual_nb_neurons["fc2"] = fc_layer_output * nbClasses

        print(net.name)
        print(manual_nb_neurons)

        return net

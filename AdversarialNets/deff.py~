
import random
import numpy as np
import pickle

from collections import deque

import tflearn
import tensorflow as tf
from tflearn import time_distributed, conv_2d

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen = buffer_size)
        self.count = len(self.buffer)
        
    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        self.buffer.append(experience)

    def save(self, path):
        pickle.dump(self.buffer, open(path, "wb"))

    def load(self, path):
        self.buffer = pickle.load(open(path, "rb"))
        print('\033[92m' + 'Buffer found with {} data points \033[0m'.format(len(self.buffer)))
        # if len(self.buffer) > self.buffer_size:
        #     print('Trimming to {}'.format(self.buffer_size))
        #     self.buffer = self.buffer[:-self.buffer_size]

    def size(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        '''
        batch_size specifies the number of experiences to add
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least
        batch_size elements before beginning to sample from it.
        '''
        batch = []

        if len(self.buffer) < batch_size:
            batch = random.sample(self.buffer, len(self.buffer))
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer = deque(maxlen = self.buffer_size)
        self.count = 0

class defActorNetwork(object):

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.saver = tf.train.Saver()
        self.sess = sess
        self.s_dim = tf.placeholder(tf.int8, state_dim)
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        self.inputs, self.out, self.scaled_out = self.create_Dactor_network()
        self.network_params = tf.trainable_variables('DEF')
        print(self.network_params)

        self.target_inputs, self.target_out, self.target_scaled_out = self.create_Dactor_network()
        self.target_network_params = tf.trainable_variables('DEF')[len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + \
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]


        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, 588])

        # Combine the gradients, dividing by the batch size to
        # account for the fact that the gradients are summed over the
        # batch by tf.gradients
        self.unnormalized_actor_gradients = tf.gradients( self.scaled_out, self.network_params, -self.action_gradient )
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        with tf.variable_scope('adam1'):
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_Dactor_network(self):
        with tf.variable_scope('DEF'):
            with tf.variable_scope('Dinputs1'):
                inputs = tflearn.input_data(shape=[None, 5, 7, 7])
            with tf.variable_scope('Dconv1'):
                conv = conv_2d(inputs, nb_filter=9, filter_size=3, strides=1, reuse=tf.AUTO_REUSE, scope='def', trainable=True)
                # print(conv.get_shape().as_list())    ---> [None, 5, 7, 9], 5*7*9 = 315
                conv = tflearn.reshape(conv, [-1, 5, 63]) # [None, 5, 315/5 ]
            with tf.variable_scope('Dlstm'):
                net = tflearn.lstm(conv, 256, return_seq=True, reuse=tf.AUTO_REUSE, scope='def', trainable=True)
                net = tflearn.dropout(net, 0.6)
                net = tflearn.activations.elu(net)
                net = tflearn.reshape(net, [-1, 5, 256])
            with tf.variable_scope('Dlstm1'):
                net = tflearn.lstm(net, 256, reuse=tf.AUTO_REUSE, scope='def')
                net = tflearn.dropout(net, 0.6)
                net = tflearn.activations.elu(net)
                w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            with tf.variable_scope('out'):
                out = tflearn.fully_connected(
                    net, 588, activation='sigmoid', weights_init=w_init, reuse=tf.AUTO_REUSE, scope='def')
                # Scale output to -action_bound to action_bound
                scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def save(self):
        print('Saving actor...')
        save_path = self.saver.save(self.sess, 'AdversarialNets/Dactor/Dactor.ckpt')

    def load(self):
        print('Restoring actor...')
        res_path = self.saver.restore(self.sess, 'AdversarialNets/Dactor/Dactor.ckpt')
        print('Actor restored...')

class defCriticNetwork(object):

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.saver = tf.train.Saver()
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        self.inputs, self.action, self.out = self.create_Dcritic_network()

        self.network_params = tf.trainable_variables('DEF')[num_actor_vars:]

        self.target_inputs, self.target_action, self.target_out = self.create_Dcritic_network()

        self.target_network_params = tf.trainable_variables('DEF')[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + \
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]
        # Network target (y_i)
        # Obtained from the target networks
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        with tf.variable_scope('adam2'):
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action
        self.action_grads = tf.gradients(self.out, self.action)

        # DE FACUT

    def create_Dcritic_network(self):
        with tf.variable_scope('DEF'):
            with tf.variable_scope('inputs2'):
                inputs = tflearn.input_data(shape=[None, 5, 7, 7])
                action = tflearn.input_data(shape=[None, 588])
            with tf.variable_scope('conv2'):
                conv = conv_2d(inputs, nb_filter=9, filter_size=3, strides=1, reuse=tf.AUTO_REUSE, scope='def')
                conv = tflearn.reshape(conv, [-1, 5, 63])

            with tf.variable_scope('lstm2'):
                net = tflearn.lstm(conv, 256, return_seq=True, reuse=tf.AUTO_REUSE, scope='def')
                net = tflearn.dropout(net, 0.6)
                net = tflearn.activations.elu(net)
            # Add the action tensor in the 2nd hidden layer
            # Use two temp layers to get the corresponding weights and biases
            net = tflearn.reshape(net, [-1, 1280])
            with tf.variable_scope('fc1'):
                t1 = tflearn.fully_connected(net, 600, reuse=tf.AUTO_REUSE, scope='def')
            with tf.variable_scope('fc2'):
                t2 = tflearn.fully_connected(action, 600, reuse=tf.AUTO_REUSE, scope='def')

            net = tflearn.activation(
                 tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='tanh')

            # linear layer connected to 1 output representing Q(s,a)
            # Weights are init to Uniform[-3e-3, 3e-3]
            w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            with tf.variable_scope('out2'):
                out = tflearn.fully_connected(net, 1, weights_init=w_init, reuse=tf.AUTO_REUSE, scope='def')
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def save(self):
        print('saving critic..')
        save_path = self.saver.save(self.sess, 'AdversarialNets/Dcritic/Dcritic.ckpt')

    def load(self):
        print('Restoring critic...')
        self.saver.restore(self.sess, 'AdversarialNets/Dcritic/Dcritic.ckpt')
        print('Critic restored...')


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

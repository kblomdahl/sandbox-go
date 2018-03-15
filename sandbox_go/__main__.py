# Copyright (c) 2018 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from sandbox_go.rules.features import NUM_FEATURES, pattern_embedding_initializer
import sandbox_go.sgf as sgf

import tensorflow as tf
import numpy as np

from datetime import datetime
from glob import glob

MAX_STEPS = 52428800  # the total number of examples to train over
BATCH_SIZE = 512  # the number of examples per batch


class EmbeddingLayer:
    """ Embeddings layer. """

    def __init__(self, channel, shape, initializer=None):
        self._channel = channel
        self._shape = shape

        self._embedding = tf.get_variable('embeddings', shape, initializer=initializer)

    def __call__(self, x, mode):
        # extract and flatten the channel that we are going to replace with an
        # embedding
        x_unstack = tf.unstack(x, axis=1)  # unstack channels
        x_ids = tf.cast(tf.reshape(x_unstack[self._channel], [-1]), tf.int32)
        x_pattern = tf.nn.embedding_lookup(
            self._embedding,
            x_ids
        )

        # since the embedding is at the last dimension, and we are using the NCHW
        # order, we need to transpose the embedded tensor
        x_pattern = tf.reshape(x_pattern, [-1, 19, 19, self._shape[1]])
        x_pattern = tf.transpose(x_pattern, [0, 3, 1, 2])

        # replace the channel in the input vector with the embeddings
        x_pattern_unstack = tf.unstack(x_pattern, axis=1)
        x_head = x_unstack[:self._channel]
        x_tail = x_unstack[(self._channel+1):]

        return tf.stack(x_head + x_pattern_unstack + x_tail, axis=1)


class BatchNorm:
    """ Batch normalization layer. """

    def __init__(self, num_channels, suffix=None):
        if not suffix:
            suffix = ''

        ones_op = tf.ones_initializer()
        zeros_op = tf.zeros_initializer()

        self._scale = tf.get_variable('scale'+suffix, (num_channels,), tf.float32, ones_op, trainable=False)
        self._offset = tf.get_variable('offset'+suffix, (num_channels,), tf.float32, zeros_op, trainable=True)
        self._mean = tf.get_variable('mean'+suffix, (num_channels,), tf.float32, zeros_op, trainable=False)
        self._variance = tf.get_variable('variance'+suffix, (num_channels,), tf.float32, ones_op, trainable=False)

    def __call__(self, x, mode):
        if mode == tf.estimator.ModeKeys.TRAIN:
            y, b_mean, b_variance = tf.nn.fused_batch_norm(
                x,
                self._scale,
                self._offset,
                None,
                None,
                data_format='NCHW',
                is_training=True
            )

            with tf.device(None):
                update_mean_op = tf.assign_sub(self._mean, 0.01 * (self._mean - b_mean), use_locking=True)
                update_variance_op = tf.assign_sub(self._variance, 0.01 * (self._variance - b_variance), use_locking=True)

                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean_op)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_variance_op)
        else:
            y, _, _ = tf.nn.fused_batch_norm(
                x,
                self._scale,
                self._offset,
                self._mean,
                self._variance,
                data_format='NCHW',
                is_training=False
            )

        return y


class ResidualBlock:
    """
    A single residual block as described by DeepMind.

    1. A convolution of 256 filters of kernel size 3 × 3 with stride 1
    2. Batch normalisation
    3. A rectifier non-linearity
    4. A convolution of 256 filters of kernel size 3 × 3 with stride 1
    5. Batch normalisation
    6. A skip connection that adds the input to the block
    7. A rectifier non-linearity
    """

    def __init__(self, params):
        init_op = tf.glorot_normal_initializer()
        num_channels = params['num_channels']

        self._conv_1 = tf.get_variable('weights_1', (3, 3, num_channels, num_channels), tf.float32, init_op)
        self._bn_1 = BatchNorm(num_channels, suffix='_1')
        self._conv_2 = tf.get_variable('weights_2', (3, 3, num_channels, num_channels), tf.float32, init_op)
        self._bn_2 = BatchNorm(num_channels, suffix='_2')

        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(self._conv_1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(self._conv_2))

    def __call__(self, x, mode):
        y = tf.nn.conv2d(x, self._conv_1, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = self._bn_1(y, mode)
        y = tf.nn.relu(y)

        y = tf.nn.conv2d(y, self._conv_2, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = self._bn_2(y, mode)
        y = tf.nn.relu(y + x)

        return y


class ValueHead:
    """
    The value head attached after the residual blocks as described by DeepMind:

    1. A convolution of 1 filter of kernel size 1 × 1 with stride 1
    2. Batch normalisation
    3. A rectifier non-linearity
    4. A fully connected linear layer to a hidden layer of size 256
    5. A rectifier non-linearity
    6. A fully connected linear layer to a scalar
    7. A tanh non-linearity outputting a scalar in the range [-1, 1]
    """

    def __init__(self, params):
        init_op = tf.orthogonal_initializer()
        zeros_op = tf.zeros_initializer()
        num_channels = params['num_channels']

        self._downsample = tf.get_variable('downsample', (1, 1, num_channels, 1), tf.float32, init_op)
        self._bn = BatchNorm(1)
        self._weights_1 = tf.get_variable('weights_1', (361, 256), tf.float32, init_op)
        self._weights_2 = tf.get_variable('weights_2', (256, 1), tf.float32, init_op)
        self._bias_1 = tf.get_variable('bias_1', (256,), tf.float32, zeros_op)
        self._bias_2 = tf.get_variable('bias_2', (1,), tf.float32, zeros_op)

    def __call__(self, x, mode):
        y = tf.nn.conv2d(x, self._downsample, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = self._bn(y, mode)
        y = tf.nn.relu(y)

        y = tf.reshape(y, (-1, 361))
        y = tf.matmul(y, self._weights_1) + self._bias_1
        y = tf.nn.relu(y)
        y = tf.matmul(y, self._weights_2) + self._bias_2

        return tf.nn.tanh(y)


class PolicyHead:
    """
    The policy head attached after the residual blocks as described by DeepMind:

    1. A convolution of 2 filters of kernel size 1 × 1 with stride 1
    2. Batch normalisation
    3. A rectifier non-linearity
    4. A fully connected linear layer that outputs a vector of size 19**2 + 1 = 362 corresponding to
       logit probabilities for all intersections and the pass move
    """

    def __init__(self, params):
        init_op = tf.orthogonal_initializer()
        zeros_op = tf.zeros_initializer()
        num_channels = params['num_channels']

        self._downsample = tf.get_variable('downsample', (1, 1, num_channels, 2), tf.float32, init_op)
        self._bn = BatchNorm(2)
        self._weights = tf.get_variable('weights', (722, 362), tf.float32, init_op)
        self._bias = tf.get_variable('bias', (362,), tf.float32, zeros_op)

    def __call__(self, x, mode):
        y = tf.nn.conv2d(x, self._downsample, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = self._bn(y, mode)
        y = tf.nn.relu(y)

        y = tf.reshape(y, (-1, 722))
        y = tf.matmul(y, self._weights) + self._bias

        return y


class Tower:
    """
    The full neural network used to predict the value and policy tensors for a mini-batch of board
    positions.
    """

    def __init__(self, params):
        init_op = tf.glorot_normal_initializer()
        num_blocks = params['num_blocks']
        num_channels = params['num_channels']
        num_patterns = params['num_patterns']
        num_inputs = (NUM_FEATURES - 1) \
            + num_patterns

        with tf.variable_scope('01_upsample'):
            self._embedding = EmbeddingLayer(2, [22665, num_patterns], initializer=pattern_embedding_initializer)
            self._upsample = tf.get_variable('weights', (3, 3, num_inputs, num_channels), tf.float32, init_op)
            self._bn = BatchNorm(num_channels)

        self._residuals = []

        for i in range(num_blocks):
            with tf.variable_scope('{:02d}_residual'.format(2 + i)):
                self._residuals += [ResidualBlock(params)]

        # policy head
        with tf.variable_scope('{:02d}p_policy'.format(2 + num_blocks)):
            self._policy = PolicyHead(params)

        # value head
        with tf.variable_scope('{:02d}v_value'.format(2 + num_blocks)):
            self._value = ValueHead(params)

    def __call__(self, x, mode):
        y = self._embedding(x, mode)
        y = tf.nn.conv2d(y, self._upsample, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        y = self._bn(y, mode)
        y = tf.nn.relu(y)

        for resb in self._residuals:
            y = resb(y, mode)

        p = self._policy(y, mode)
        v = self._value(y, mode)

        return v, p


def get_dataset():
    def _parse_sgf(line):
        try:
            return sgf.one(line)
        except ValueError:  # bad game
            return (
                np.asarray([], 'f4'),  # features
                np.asarray([0.0], 'f4'),  # value
                np.asarray([], 'f4'),  # policy
            )

    def _is_valid(_features, value, _policy):
        return tf.reduce_any(tf.not_equal(value, 0.0))

    def _fix_shape(features, value, policy):
        features = tf.reshape(features, [NUM_FEATURES, 19, 19])
        value = tf.reshape(value, [1])
        policy = tf.reshape(policy, [362])

        return features, value, policy

    dataset = tf.data.TextLineDataset(glob('data/*.sgf'))
    dataset = dataset.repeat()
    dataset = dataset.map(lambda text: tuple(tf.py_func(
        _parse_sgf,
        [text],
        [tf.float32, tf.float32, tf.float32]
    )))
    dataset = dataset.filter(_is_valid)
    dataset = dataset.map(_fix_shape)
    dataset = dataset.shuffle(384000)
    dataset = dataset.batch(BATCH_SIZE)

    return dataset


def input_fn():
    return get_dataset().map(lambda features, value, policy:
        (features, {'value': value, 'policy': policy})
    )


def model_fn(features, labels, mode, params):
    global_step = tf.train.get_global_step()
    tower = Tower(params)
    value_hat, policy_hat = tower(features, mode)

    # determine the loss for each of the components:
    #
    # - L2 regularization
    # - Value head
    # - Policy head
    #
    loss_l2 = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss_value = tf.reduce_mean(tf.squared_difference(
        tf.stop_gradient(labels['value']),
        value_hat
    ))
    loss_policy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=tf.stop_gradient(labels['policy']),
        logits=policy_hat
    ))

    loss = loss_policy + loss_value + 8e-4 * loss_l2

    # setup the optimizer to use a constant learning rate of `0.1` for the
    # first 30% of the steps, then use an exponential decay. This is similar to
    # cosine decay, and has proven critical to the value head converging at
    # all.
    learning_steps = MAX_STEPS//BATCH_SIZE
    learning_rate_threshold = int(0.3 * learning_steps)
    learning_rate_exp = tf.train.exponential_decay(
        0.01,
        global_step - learning_rate_threshold,
        (learning_steps - learning_rate_threshold) / 200,
        0.98
    )

    learning_rate = tf.train.piecewise_constant(
        global_step,
        [learning_rate_threshold],
        [0.01, learning_rate_exp]
    )
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step)

    # setup some nice looking metric to look at
    if mode == tf.estimator.ModeKeys.TRAIN:
        policy_hot = tf.argmax(labels['policy'], axis=1)

        def _in_top_k(k):
            in_top_k = tf.nn.in_top_k(policy_hat, policy_hot, k=k)

            return tf.reduce_mean(tf.cast(in_top_k, tf.float32))

        def _sign_equal():
            same_sign = tf.equal(tf.sign(labels['value']), tf.sign(value_hat))

            return tf.reduce_mean(tf.cast(same_sign, tf.float32))

        tf.summary.scalar('accuracy/policy_1', _in_top_k(1))
        tf.summary.scalar('accuracy/policy_3', _in_top_k(3))
        tf.summary.scalar('accuracy/policy_5', _in_top_k(5))
        tf.summary.scalar('accuracy/value', _sign_equal())

        tf.summary.scalar('loss/policy', loss_policy)
        tf.summary.scalar('loss/value', loss_value)
        tf.summary.scalar('loss/l2', loss_l2)

        tf.summary.scalar('learning_rate', learning_rate)

    # put it all together into a specification
    return tf.estimator.EstimatorSpec(
        mode,
        {'value': value_hat, 'policy': tf.nn.softmax(policy_hat)},
        loss,
        train_op,
        {},  # eval_metric_ops
    )

# reduce the amount of spam that we're getting to the console
tf.logging.set_verbosity(tf.logging.WARN)

config = tf.estimator.RunConfig(
    session_config = tf.ConfigProto(
        gpu_options = tf.GPUOptions(
            allow_growth = True
        )
    )
)

nn = tf.estimator.Estimator(
    config=config,
    model_fn=model_fn,
    model_dir='models/' + datetime.now().strftime('%Y%m%d.%H%M') + '-p08/',
    params={'num_channels': 128, 'num_patterns': 8, 'num_blocks': 9}
)
nn.train(input_fn=input_fn, steps=MAX_STEPS//BATCH_SIZE)

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

from sandbox_go.rules.features import NUM_FEATURES
import sandbox_go.sgf as sgf

import tensorflow as tf
import numpy as np

from datetime import datetime
from glob import glob
from math import sqrt

MAX_STEPS = 52428800  # the total number of examples to train over
BATCH_SIZE = 512  # the number of examples per batch
LSUV_OPS = 'LSUVOps'  # the graph collection that contains all lsuv operations

def orthogonal_initializer():
    """ Returns an orthogonal initializer that use QR-factorization to find
    the orthogonal basis of a random matrix. This differs from the Tensorflow
    implementation in that it checks for singular matrices, which is mostly a
    problem when generating small matrices. """

    def _init(shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = tf.float32

        assert len(shape) >= 2

        # flatten the input shape with the last dimension remaining so it works
        # for convolutions
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]

        flat_shape = (num_cols, num_rows) if num_rows < num_cols else (num_rows, num_cols)

        # check so that the random matrix is not singular
        while True:
            a = np.random.standard_normal(flat_shape)
            q, r = np.linalg.qr(a)
            d = np.diag(r)

            if np.prod(d) > 1e-2:
                break

        ph = d / np.abs(d)
        q *= ph

        if num_rows < num_cols:
            q = np.transpose(q, [1, 0])

        return np.reshape(q, shape)

    return _init


def lsuv_initializer(output, weights):
    """ Returns an operation that initialize the given weights and their output
    using the LSUV [1] methodology.

    [1] Dmytro Mishkin, Jiri Matas, "All you need is a good init" """

    _, variance = tf.nn.moments(output, axes=[0, 2, 3], keep_dims=True)
    variance = tf.transpose(variance, [0, 2, 3, 1])

    update_op = tf.assign(weights, tf.truediv(weights, tf.sqrt(variance)), use_locking=True)

    with tf.control_dependencies([update_op]):
        name = weights.name.split(':')[0] + '/lsuv'

        return tf.sqrt(variance, name=name)

class LSUVInit(tf.train.SessionRunHook):
    """ LSUV [1] initialization hook that calls any operations added to
    the `LSUV_OPS` graph collection twice in sequence.

    [1] Dmytro Mishkin, Jiri Matas, "All you need is a good init" """

    def before_run(self, run_context):
        session = run_context.session
        global_step = tf.train.get_global_step()
        if global_step.eval(session) > 0:
            return

        count = 0

        for lsuv_op in tf.get_collection(LSUV_OPS):
            for _ in range(2):
                _std = session.run([lsuv_op])

            count += 1

        print('LSUV initialization finished, adjusted %d tensors.' % (count,))

class EmbeddingLayer:
    """ Embeddings layer. """

    def __init__(self, channel, shape, initializer=None, suffix=None):
        if not suffix:
            suffix = ''

        self._channel = channel
        self._shape = shape

        self._embedding = tf.get_variable('embeddings' + suffix, shape, initializer=initializer)

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

        return tf.stack(x_head + x_tail + x_pattern_unstack, axis=1)


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
        init_op = orthogonal_initializer()
        num_channels = params['num_channels']
        num_blocks = params['num_blocks']

        self._conv_1 = tf.get_variable('weights_1', (3, 3, num_channels, num_channels), tf.float32, init_op)
        self._bn_1 = BatchNorm(num_channels, suffix='_1')
        self._conv_2 = tf.get_variable('weights_2', (3, 3, num_channels, num_channels), tf.float32, init_op)
        self._bn_2 = BatchNorm(num_channels, suffix='_2')
        self._scale = 1.0 / sqrt(num_blocks)

        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(self._conv_1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(self._conv_2))

    def __call__(self, x, mode):
        y = tf.nn.conv2d(x, self._conv_1, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        tf.add_to_collection(LSUV_OPS, lsuv_initializer(y, self._conv_1))

        y = self._bn_1(y, mode)
        y = tf.nn.relu(y)

        y = tf.nn.conv2d(y, self._conv_2, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        tf.add_to_collection(LSUV_OPS, lsuv_initializer(y, self._conv_2))

        y = self._bn_2(y, mode)
        y = tf.nn.relu(self._scale * y + x)

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
        init_op = orthogonal_initializer()
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
        tf.add_to_collection(LSUV_OPS, lsuv_initializer(y, self._downsample))

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
        init_op = orthogonal_initializer()
        zeros_op = tf.zeros_initializer()
        num_channels = params['num_channels']

        self._downsample = tf.get_variable('downsample', (1, 1, num_channels, 2), tf.float32, init_op)
        self._bn = BatchNorm(2)
        self._weights = tf.get_variable('weights', (722, 362), tf.float32, init_op)
        self._bias = tf.get_variable('bias', (362,), tf.float32, zeros_op)

    def __call__(self, x, mode):
        y = tf.nn.conv2d(x, self._downsample, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        tf.add_to_collection(LSUV_OPS, lsuv_initializer(y, self._downsample))

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
        init_op = orthogonal_initializer()
        num_blocks = params['num_blocks']
        num_channels = params['num_channels']
        num_patterns = params['num_patterns']
        num_inputs = (NUM_FEATURES - 2) \
            + 2 * num_patterns

        with tf.variable_scope('01_upsample'):
            self._embedding1 = EmbeddingLayer(2, [22665, num_patterns], initializer=init_op, suffix='_1')
            self._embedding2 = EmbeddingLayer(2, [22666, num_patterns], initializer=init_op, suffix='_2')
            self._upsample = tf.get_variable('weights', (3, 3, num_inputs, num_channels), tf.float32, init_op)
            self._bn = BatchNorm(num_channels)

        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(self._upsample))

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
        y = self._embedding1(x, mode)
        y = self._embedding2(y, mode)
        y = tf.nn.conv2d(y, self._upsample, (1, 1, 1, 1), 'SAME', True, 'NCHW')
        tf.add_to_collection(LSUV_OPS, lsuv_initializer(y, self._upsample))

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
                np.asarray([], 'f4'),  # value
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

    # setup the optimizer to use a constant learning rate of `0.01` for the
    # first 30% of the steps, then use an exponential decay. This is similar to
    # cosine decay, and has proven critical to the value head converging at
    # all.
    # 
    # We then clip the gradients by its global norm to avoid some gradient
    # explosions that seems to occur during the first few steps.
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
        gradients, variables = zip(*optimizer.compute_gradients(
            loss,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
            colocate_gradients_with_ops=True
        ))

        clip_gradients, global_norm = tf.clip_by_global_norm(gradients, 5.0)
        train_op = optimizer.apply_gradients(zip(clip_gradients, variables), global_step)

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

        tf.summary.scalar('gradients/global_norm', global_norm)

        for grad, var in zip(gradients, variables):
            var_name = var.name.split(':', 2)[0]

            tf.summary.scalar('gradients/' + var_name, tf.norm(grad))
            tf.summary.scalar('norms/' + var_name, tf.norm(var))

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
    model_dir='models/' + datetime.now().strftime('%Y%m%d.%H%M') + '-p08-orthogonal-next/',
    params={'num_channels': 128, 'num_patterns': 8, 'num_blocks': 9}
)
nn.train(input_fn=input_fn, hooks=[LSUVInit()], steps=MAX_STEPS//BATCH_SIZE)

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

def embedding_layer(x, shape, channel, name=None):
    with tf.variable_scope(name, 'embedding'):
        embeddings = tf.get_variable('embeddings', shape)

        # extract and flatten the channel that we are going to replace with an
        # embedding
        x_unstack = tf.unstack(x, axis=1)  # unstack channels
        x_ids = tf.cast(tf.reshape(x_unstack[channel], [-1]), tf.int32)

        x_pattern = tf.nn.embedding_lookup(
            embeddings,
            x_ids,
            max_norm=shape[1]
        )

        # since the embedding is at the last dimension, and we are using the NCHW
        # order, we need to transpose the embedded tensor
        x_pattern = tf.reshape(x_pattern, [-1, 19, 19, shape[1]])
        x_pattern = tf.transpose(x_pattern, [0, 3, 1, 2])

        # replace the channel in the input vector with the embeddings
        x_pattern_unstack = tf.unstack(x_pattern, axis=1)
        x_head = x_unstack[:channel]
        x_tail = x_unstack[(channel+1):]

        return tf.stack(x_head + x_pattern_unstack + x_tail, axis=1)


def prelu(x):
    """ Parameterised relu. """

    with tf.variable_scope('prelu'):
        alpha = tf.get_variable('alpha')

        return tf.nn.leaky_relu(x, alpha)


def tower(x, mode, params):
    y = embedding_layer(x, [22665, params['num_patterns']], 2, name='pattern')

    # the start of the tower as described by DeepMind:
    #
    # 1. A convolution of 256 filters of kernel size 3×3 with stride 1
    # 2. Batch normalization
    # 3. A rectifier nonlinearity
    #
    y = tf.layers.conv2d(
        y,
        params['num_channels'],  # filters
        3,  # kernel_size
        1,  # strides
        'same',  # padding
        'channels_first',  # data_format
        use_bias=True,
        bias_initializer=tf.random_uniform_initializer(-1.0, 1.0),
        kernel_initializer=tf.glorot_normal_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0)
    )

    y = tf.layers.batch_normalization(
        y,
        axis=1,  # data_format
        center=False,
        scale=False,
        fused=True,
        training=(mode == tf.estimator.ModeKeys.TRAIN),
        trainable=False
    )

    y = tf.nn.relu(y)

    # The residual blocks as described by DeepMind:
    #
    #   1. A convolution of 256 filters of kernel size 3×3 with stride 1
    #   2. Batch normalization
    #   3. A rectifier nonlinearity
    #   4. A convolution of 256 filters of kernel size 3×3 with stride 1
    #   5. Batch normalization
    #   6. A skip connection that adds the input to the block
    #   7. A rectifier nonlinearity
    #
    for _ in range(params['num_blocks']):
        z = tf.layers.conv2d(
            y,
            params['num_channels'],  # filters
            3,  # kernel_size
            1,  # strides
            'same',  # padding
            'channels_first',  # data_format
            use_bias=True,
            bias_initializer=tf.random_uniform_initializer(-1.0, 1.0),
            kernel_initializer=tf.glorot_normal_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0)
        )

        z = tf.layers.batch_normalization(
            z,
            axis=1,  # data_format
            center=False,
            scale=False,
            fused=True,
            training=(mode == tf.estimator.ModeKeys.TRAIN),
            trainable=False
        )

        z = tf.nn.relu(z)

        z = tf.layers.conv2d(
            z,
            params['num_channels'],  # filters
            3,  # kernel_size
            1,  # strides
            'same',  # padding
            'channels_first',  # data_format
            use_bias=True,
            bias_initializer=tf.random_uniform_initializer(-1.0, 1.0),
            kernel_initializer=tf.glorot_normal_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0)
        )

        z = tf.layers.batch_normalization(
            z,
            axis=1,  # data_format
            center=False,
            scale=False,
            fused=True,
            training=(mode == tf.estimator.ModeKeys.TRAIN),
            trainable=False
        )

        y = tf.nn.relu(z + y)
        del z

    # The policy head as described by DeepMind:
    #
    #   1. A convolution of 2 filters of kernel size 1×1 with stride 1
    #   2. Batch normalization
    #   3. A rectifier nonlinearity
    #   4. A fully connected linear layer that outputs a vector of
    #      size 19² + 1 = 362.
    #
    p = tf.layers.conv2d(
        y,
        2,  # filters
        1,  # kernel_size
        1,  # strides
        'same',  # padding
        'channels_first',  # data_format
        use_bias=True,
        bias_initializer=tf.random_uniform_initializer(-1.0, 1.0),
        kernel_initializer=tf.glorot_normal_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0)
    )

    p = tf.layers.batch_normalization(
        p,
        axis=1,  # data_format
        center=False,
        scale=False,
        fused=True,
        training=(mode == tf.estimator.ModeKeys.TRAIN),
        trainable=False
    )

    p = tf.nn.relu(p)

    p = tf.layers.dense(
        tf.reshape(p, [-1, 722]),  # inputs
        362,  # units
        use_bias=True,
        bias_initializer=tf.random_uniform_initializer(-1.0, 1.0),
        kernel_initializer=tf.orthogonal_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0)
    )

    # The value head as described by DeepMind:
    #
    #   1. A convolution of 1 filter of kernel size 1×1 with stride 1
    #   2. Batch normalization
    #   3. A rectifier nonlinearity
    #   4. A fully connected linear layer to a hidden layer of size 256
    #   5. A rectifier nonlinearity
    #   6. A fully connected linear layer to a scalar
    #   7. A tanh nonlinearity outputting a scalar in the range [−1, 1]
    #
    v = tf.layers.conv2d(
        y,  # inputs
        1,  # filters
        1,  # kernel_size
        1,  # strides
        'same',  # padding
        'channels_first',  # data_format
        use_bias=True,
        bias_initializer=tf.random_uniform_initializer(-1.0, 1.0),
        kernel_initializer=tf.glorot_normal_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0)
    )

    v = tf.layers.batch_normalization(
        v,
        axis=1,  # data_format
        center=False,
        scale=False,
        fused=True,
        training=(mode == tf.estimator.ModeKeys.TRAIN),
        trainable=False
    )

    v = tf.nn.relu(v)

    v = tf.layers.dense(
        tf.reshape(v, [-1, 361]),  # inputs
        256,  # units
        use_bias=True,
        bias_initializer=tf.random_uniform_initializer(-1.0, 1.0),
        kernel_initializer=tf.orthogonal_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0)
    )

    v = tf.nn.relu(v)

    v = tf.layers.dense(
        v,  # inputs
        1,  # units
        use_bias=True,
        bias_initializer=tf.random_uniform_initializer(-1.0, 1.0),
        kernel_initializer=tf.orthogonal_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0)
    )

    v = tf.nn.tanh(v)

    return v, p


def get_dataset(batch_size):
    def _parse_sgf(line):
        try:
            return sgf.one(line)
        except ValueError:  # bad game
            return (
                np.asarray([], 'f4'),  # features
                np.asarray([0.0], 'f4'),  # value
                np.asarray([], 'f4'),  # policy
            )

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
    dataset = dataset.filter(lambda _f, value, _p1: tf.not_equal(value, 0.0))
    dataset = dataset.map(_fix_shape)
    dataset = dataset.shuffle(384000)
    dataset = dataset.batch(batch_size)

    return dataset


def input_fn(batch_size):
    return get_dataset(batch_size).map(lambda features, value, policy:
        (features, {'value': value, 'policy': policy})
    )


def model_fn(features, labels, mode, params):
    value_hat, policy_hat = tower(features, mode, params)

    # determine the loss
    loss_l2 = tf.losses.get_regularization_loss()
    loss_value =  tf.losses.mean_squared_error(
        labels['value'],
        value_hat,
        weights=1.0
    )
    loss_policy = tf.losses.softmax_cross_entropy(
        labels['policy'],
        policy_hat,
        weights=1.0
    )

    loss = loss_policy + loss_value + 8e-4 * loss_l2

    # setup the optimizer
    global_step = tf.train.get_global_step()
    learning_rate = tf.train.exponential_decay(1e-1, global_step, 16000 / params['batch_size'], 0.96)
    optimizer = tf.train.MomentumOptimizer(0.1, learning_rate, use_nesterov=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step)

    # setup some nice looking metric to look at
    if mode == tf.estimator.ModeKeys.TRAIN:
        policy_hot = tf.argmax(labels['policy'], axis=1)

        tf.summary.scalar('accuracy/policy_1', tf.reduce_mean(tf.cast(tf.nn.in_top_k(policy_hat, policy_hot, k=1), tf.float32)))
        tf.summary.scalar('accuracy/policy_3', tf.reduce_mean(tf.cast(tf.nn.in_top_k(policy_hat, policy_hot, k=3), tf.float32)))
        tf.summary.scalar('accuracy/policy_5', tf.reduce_mean(tf.cast(tf.nn.in_top_k(policy_hat, policy_hot, k=5), tf.float32)))
        tf.summary.scalar('accuracy/value', tf.reduce_mean(tf.cast(tf.equal(tf.sign(labels['value']), tf.sign(value_hat)), tf.float32)))

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

batch_size = 256
nn = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir='models/' + datetime.now().strftime('%Y%m%d.%H%M') + '/',
    params={'num_channels': 128, 'num_blocks': 9, 'num_patterns': 32, 'batch_size': batch_size}
)
nn.train(input_fn=lambda: input_fn(batch_size), steps=26214400/batch_size)

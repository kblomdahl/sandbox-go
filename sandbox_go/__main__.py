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

TILES = [1, 4, 7, 9, 11, 14, 17]

def tower(x, mode, params):
    with tf.variable_scope('mini'):
        p_embedding = tf.get_variable('embeddings', [22665, params['num_patterns']])
        x_ids = tf.cast(tf.reshape(x, [-1]), tf.int32)

    y = tf.nn.embedding_lookup(p_embedding, x_ids, max_norm=params['num_patterns'])
    y = tf.reshape(y, [-1, 9 * params['num_patterns']])

    y = tf.layers.dense(
        y,  # inputs
        10,  # units
        use_bias=False,
        kernel_initializer=tf.orthogonal_initializer()
    )

    # re-construct all of the sub-tiles into the full board, by first figuring
    # out what tiles should go to what index in the policy, we then either:
    #
    # - average if two tiles overlap
    # - minimum for the pass move
    #
    policy_indices = [None] * 361
    policy_slices = [None] * 362
    y = tf.reshape(y, [-1, 49, 10])

    for sy in range(7):
        for sx in range(7):
            tile = 7 * sy + sx
            tindex = 0

            for yy in range(TILES[sy] - 1, TILES[sy] + 2):
                for xx in range(TILES[sx] - 1, TILES[sx] + 2):
                    index = 19 * yy + xx

                    if not policy_indices[index]:
                        policy_indices[index] = []

                    policy_indices[index].append(y[:, tile, tindex])
                    tindex += 1

    for i in range(361):
        policy_slices[i] = tf.reduce_mean(policy_indices[i], axis=0)
    policy_slices[361] = tf.reduce_min(y[:,:,9], axis=1)

    for i in range(362):
        policy_slices[i] = tf.reshape(policy_slices[i], [-1, 1])

    return tf.concat(policy_slices, axis=1)


def get_dataset(batch_size):
    def _parse_sgf(line):
        try:
            features, policy =  sgf.one(line)

            return features, policy
        except ValueError:  # bad game
            return (
                np.zeros((49, 9), 'f4'),
                np.zeros((362,), 'f4'),
            )

    dataset = tf.data.TextLineDataset(glob('data/*.sgf'))
    dataset = dataset.repeat()
    dataset = dataset.map(lambda text: tuple(tf.py_func(
        _parse_sgf,
        [text],
        [tf.float32, tf.float32]
    )))
    dataset = dataset.filter(lambda features, policy: tf.reduce_any(tf.not_equal(policy, 0.0)))
    dataset = dataset.shuffle(1176000)
    dataset = dataset.batch(batch_size)

    return dataset


def input_fn(batch_size):
    return get_dataset(batch_size).map(lambda features, policy:
        (features, {'policy': policy})
    )


def model_fn(features, labels, mode, params):
    policy_hat = tower(features, mode, params)
    policy_hot = tf.argmax(labels['policy'], axis=1)

    # determine the loss, we set the weight of the `pass` move much lower than
    # real moves here to prevent the engine from just always predicting `pass`
    # and being happy with it.
    loss = tf.losses.softmax_cross_entropy(
        labels['policy'],
        policy_hat
    )

    # setup the optimizer
    global_step = tf.train.get_global_step()
    learning_rate = tf.train.exponential_decay(1e-1, global_step, (26214400 / params['batch_size']) / 256, 0.96)
    optimizer = tf.train.MomentumOptimizer(0.1, learning_rate, use_nesterov=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step)

    # setup some nice looking metric to look at
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar('accuracy/policy_1', tf.reduce_mean(tf.cast(tf.nn.in_top_k(policy_hat, policy_hot, k=1), tf.float32)))
        tf.summary.scalar('accuracy/policy_3', tf.reduce_mean(tf.cast(tf.nn.in_top_k(policy_hat, policy_hot, k=3), tf.float32)))
        tf.summary.scalar('accuracy/policy_5', tf.reduce_mean(tf.cast(tf.nn.in_top_k(policy_hat, policy_hot, k=5), tf.float32)))
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)

        tf.summary.histogram('policy_hot', policy_hot)
        tf.summary.histogram('policy_hat', policy_hat)

    # put it all together into a specification
    return tf.estimator.EstimatorSpec(
        mode,
        {'policy': tf.nn.softmax(policy_hat)},
        loss,
        train_op,
        {},  # eval_metric_ops
    )

# reduce the amount of spam that we're getting to the console
tf.logging.set_verbosity(tf.logging.WARN)

batch_size = 512
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
    model_dir='models/' + datetime.now().strftime('%Y%m%d.%H%M') + '/',
    params={'num_patterns': 2, 'batch_size': batch_size}
)
nn.train(input_fn=lambda: input_fn(batch_size), steps=26214400/batch_size)

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

MAX_STEPS = 52428800  # the total number of examples to train over
BATCH_SIZE = 512  # the number of examples per batch
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

def get_dataset():
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
    dataset = dataset.batch(BATCH_SIZE)

    return dataset


def input_fn(batch_size):
    return get_dataset().map(lambda features, policy:
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

    # setup the optimizer to use a constant learning rate of `0.01` for the
    # first 30% of the steps, then use an exponential decay. This is similar to
    # cosine decay, and has proven critical to the value head converging at
    # all.
    # 
    # We then clip the gradients by its global norm to avoid some gradient
    # explosions that seems to occur during the first few steps.
    global_step = tf.train.get_global_step()
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
        tf.summary.scalar('accuracy/policy_1', tf.reduce_mean(tf.cast(tf.nn.in_top_k(policy_hat, policy_hot, k=1), tf.float32)))
        tf.summary.scalar('accuracy/policy_3', tf.reduce_mean(tf.cast(tf.nn.in_top_k(policy_hat, policy_hot, k=3), tf.float32)))
        tf.summary.scalar('accuracy/policy_5', tf.reduce_mean(tf.cast(tf.nn.in_top_k(policy_hat, policy_hot, k=5), tf.float32)))
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('gradients/global_norm', global_norm)

        for grad, var in zip(gradients, variables):
            var_name = var.name.split(':', 2)[0]

            tf.summary.scalar('gradients/' + var_name, tf.norm(grad))
            tf.summary.scalar('norms/' + var_name, tf.norm(var))

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
    params={'num_patterns': 2}
)
nn.train(input_fn=input_fn, hooks=[], steps=MAX_STEPS//BATCH_SIZE)

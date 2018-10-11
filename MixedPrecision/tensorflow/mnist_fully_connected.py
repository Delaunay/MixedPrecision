# Credit to https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/#tensorflow

import tensorflow as tf
import numpy as np
import time

from MixedPrecision.tools.stats import StatStream


def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
    """
        Custom variable getter that forces trainable variables to be stored in
        float32 precision and then casts them to the training precision.
    """
    storage_dtype = tf.float32 if trainable else dtype

    variable = getter(
        name,
        shape,
        dtype=storage_dtype,
        initializer=initializer,
        regularizer=regularizer,
        trainable=trainable,
        *args,
        **kwargs
    )

    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)

    return variable


def gradients_with_loss_scaling(loss, variables, loss_scale):
    """
        Gradient calculation with loss scaling to improve numerical stability when training with float16.
    """
    return [grad / loss_scale for grad in tf.gradients(loss * loss_scale, variables)]


def create_simple_model(nbatch, nin, nout, nclass, dtype):
    """
        A simple softmax model.
    """

    data = tf.placeholder(dtype, shape=(nbatch, nin))

    with tf.name_scope('input_layer'):
        weights = tf.get_variable('iweights', (nin, nout), dtype)
        biases  = tf.get_variable('ibiases',        nout,  dtype, initializer=tf.zeros_initializer())
        hidden  = tf.nn.relu(tf.matmul(data, weights) + biases)

    with tf.name_scope('output_layer'):
        weights = tf.get_variable('oweights', (nout, nclass), dtype)
        biases  = tf.get_variable('obiases',         nclass,  dtype, initializer=tf.zeros_initializer())
        logits  = tf.matmul(hidden, weights) + biases

    target  = tf.placeholder(tf.float32, shape=(nbatch, nclass))

    # Note: The softmax should be computed in float32 precision
    loss    = tf.losses.softmax_cross_entropy(target, tf.cast(logits, tf.float32))

    return data, target, loss


def train(args, dataset):
    shape = args.shape
    nbatch = args.batch_size
    nin = shape[0] * shape[1] * shape[2]
    nout = args.hidden_size
    learning_rate = args.lr
    momentum = args.momentum
    loss_scale = args.static_loss_scale
    dtype = tf.float16 if args.half else tf.float32

    tf.set_random_seed(0)
    np.random.seed(0)

    device = '/gpu:0' if args.gpu else '/cpu'

    # Create training graph
    # ------------------------------------------------------------------------------------------------------------------
    with tf.device(device), \
         tf.variable_scope(
             # Note: This forces trainable variables to be stored as float32
             'fp32_storage', custom_getter=float32_variable_storage_getter):

        data, target, loss = create_simple_model(nbatch, nin, nout, 10, dtype)

        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # Note: Loss scaling can improve numerical stability for fp16 training
        grads = gradients_with_loss_scaling(loss, variables, loss_scale)

        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)

        training_step_op = optimizer.apply_gradients(zip(grads, variables))

        init_op = tf.global_variables_initializer()
    # ------------------------------------------------------------------------------------------------------------------

    # Run training
    sess = tf.Session()
    sess.run(init_op)

    compute_time = StatStream(1)
    floss = float('inf')

    for epoch in range(0, args.epochs):
        cstart = time.time()

        for batch in dataset:
            x, y = batch

            floss, _ = sess.run([loss, training_step_op], feed_dict={data: x, target: y})

        cend = time.time()
        compute_time += cend - cstart

        print('[{:4d}] Compute Time (avg: {:.4f}, sd: {:.4f}) Loss: {:.4f}'.format(
            1 + epoch, compute_time.avg, compute_time.sd, floss))


def load_data(args, nin):
    def generate_x():
        return np.random.normal(size=(args.batch_size, nin)).astype(np.float16)

    def generate_y():
        return np.zeros((args.batch_size, 10), dtype=np.float32)

    return [(generate_x(),generate_y()) for i in range(args.epochs)]


def main():
    from MixedPrecision.tools.args import get_parser

    parser = get_parser()
    args = parser.parse_args()

    for k, v in vars(args).items():
        print('{:>30}: {}'.format(k, v))

    shape = args.shape
    nin = shape[0] * shape[1] * shape[2]

    train(args, load_data(args, nin))


if __name__ == '__main__':
    main()

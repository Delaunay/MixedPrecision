# Credit to https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/#tensorflow

import tensorflow as tf
import numpy as np
import time

from MixedPrecision.tools.stats import StatStream
from MixedPrecision.tools.tensorflow import gradients_with_loss_scaling
from MixedPrecision.tools.tensorflow import float32_variable_storage_getter


"""
self.conv1 = nn.Conv2d(in_channels=self.input_shape[0], out_channels=conv_num, kernel_size=kernel_size,
                       stride=self.stride, padding=self.padding, dilation=self.dilation)

self.conv2 = nn.Conv2d(in_channels=conv_num, out_channels=conv_out, kernel_size=kernel_size,
                       stride=self.stride, padding=self.padding, dilation=self.dilation)

size = conv2d_output_size(self.conv2, self.input_shape)
print(size)
self.conv_output_size = size[1] * size[2] * size[3]
print(self.conv_output_size)
self.output_layer = nn.Linear(self.conv_output_size, 10)
"""

def create_simple_model_2(nbatch, shape, conv_num=64, conv_out=512, nclass=10, dtype=tf.float32):
    """
        A simple softmax model.
    """
    c, h, w = shape
    data = tf.placeholder(dtype, shape=(nbatch, h, w, c))

    with tf.name_scope('input_layer'):

        conv1 = tf.layers.conv2d(
            inputs=data,
            filters=conv_num,
            kernel_size=[3, 3],
            padding='SAME'
        )

    with tf.name_scope('hidden_layer'):
        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=conv_out,
            kernel_size=[3, 3],
            padding='SAME'
        )
        flat = tf.layers.Flatten()(conv2)

    with tf.name_scope('output_layer'):
        logits = tf.layers.dense(inputs=flat, units=10, use_bias=True)

    target  = tf.placeholder(tf.float32, shape=(nbatch, nclass))

    # Note: The softmax should be computed in float32 precision
    loss    = tf.losses.softmax_cross_entropy(target, tf.cast(logits, tf.float32))

    return data, target, loss


def create_simple_model(nbatch, shape, conv_num=64, conv_out=512, nclass=10, dtype=tf.float32):
    """
        A simple softmax model.
    """
    c, h, w = shape
    data = tf.placeholder(dtype, shape=(nbatch, h, w, c))

    with tf.name_scope('input_layer'):
        kernel = tf.get_variable('ikernel', filters=conv_num, shape=[3, 3, 3])
        conv = tf.nn.conv2d(data, kernel, strides=[1, 1, 1, 1], use_cudnn_on_gpu=True, data_format='NHWC', padding='SAME')
        biases = tf.get_variable('ibiases', [conv_num], tf.constant_initializer(0.0))
        ilayer = tf.nn.bias_add(conv, biases)

    with tf.name_scope('hidden_layer'):
        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=conv_out,
            kernel_size=[3, 3],
            padding='SAME'
        )
        flat = tf.layers.Flatten()(conv2)

    with tf.name_scope('output_layer'):
        logits = tf.layers.dense(inputs=flat, units=10, use_bias=True)

    target  = tf.placeholder(tf.float32, shape=(nbatch, nclass))

    # Note: The softmax should be computed in float32 precision
    loss    = tf.losses.softmax_cross_entropy(target, tf.cast(logits, tf.float32))

    return data, target, loss


def train(args, dataset):
    shape = args.shape
    nbatch = args.batch_size
    nin = shape[0] * shape[1] * shape[2]
    conv_num = args.conv_num
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

        data, target, loss = create_simple_model(nbatch, shape, conv_num, 512, 10, dtype)

        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # Note: Loss scaling can improve numerical stability for fp16 training
        grads = gradients_with_loss_scaling(loss, variables, loss_scale)

        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)

        training_step_op = optimizer.apply_gradients(zip(grads, variables))

        init_op = tf.global_variables_initializer()
    # ------------------------------------------------------------------------------------------------------------------

    # Run training
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=args.log_device_placement))
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


def load_data(args, shape):
    def generate_x():
        return np.random.normal(size=(args.batch_size, shape[1], shape[2], shape[0])).astype(np.float16)

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

    train(args, load_data(args, args.shape))


if __name__ == '__main__':
    main()

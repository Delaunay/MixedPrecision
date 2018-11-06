import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', type=str, metavar='DIR',
                        help='path to the dataset location')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--hidden_size', default=64, type=int, metavar='HS',
                        help='Size of the Hidden layer for MNIST')

    parser.add_argument('--hidden_num', default=1, type=int, metavar='HN',
                        help='Number of Hidden Layer for MNIST')

    parser.add_argument('--kernel_size', default=3, type=int, metavar='KS',
                        help='Kernel Size for Conv MNIST')

    parser.add_argument('--conv_num', default=32, type=int, metavar='CN',
                        help='Number of Conv Layer for COnv MNIST')

    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')

    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')

    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')

    parser.add_argument('--half', action='store_true',
                        help='Run model in fp16 mode.')

    parser.add_argument('--gpu', action='store_true',
                        help='Run model on gpu')

    parser.add_argument('--static-loss-scale', type=float, default=1,
                        help='Static loss scale, positive power of 2 values can improve fp16 convergence.')

    parser.add_argument('--dynamic-loss-scale', action='store_true',
                        help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                        '--static-loss-scale.')

    parser.add_argument('--prof', dest='prof', type=int, default=None,
                        help='Only run N iterations for profiling.')

    parser.add_argument('--permute', dest='permute', action='store_true', default=False,
                        help='Try to permute the tensor to use NHWC instead of NCHW')

    parser.add_argument('--fake', dest='fake', action='store_true', default=False,
                        help='Generate Random Images')

    parser.add_argument('--shape', dest='shape', type=int, nargs='*', default=[1, 28, 28],
                        help='Shape of the randomly generated images')

    parser.add_argument('--log_device_placement', action='store_true', default=False,
                        help='Make Tensorflow log device placement')

    parser.add_argument('--no-bench-mode', action='store_true', default=False,
                        help='disable benchmark mode for cudnn')

    parser.add_argument('--use-dali', action='store_true', default=False,
                        help='use dali for data loading and pre processing')

    parser.add_argument('--batch-reuse', action='store_true', default=False,
                        help='Re use old batch if data loading is slow')

    parser.add_argument('--report', type=str, default=None,
                        help='File Name to write the report in')

    parser.add_argument('--accimage', action='store_true', default=False,
                        help='High performance image loading')

    parser.add_argument('--warmup', action='store_true', default=False,
                        help='do a pre run for benchmarks')

    parser.add_argument('--benzina', action='store_true', default=False,
                        help='Use benzina')

    return parser

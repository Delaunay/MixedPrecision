#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Tuple


def conv2d_output_size(conv, i: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    co = conv.out_channels
    n = i[0]
    ho = 0
    wo = 0

    for dim in range(0, 2):
        p = conv.padding[dim]
        s = conv.stride[dim]
        k = conv.kernel_size[dim]
        d = conv.dilation[dim]

        w = i[dim + 1]
        v = (w + 2 * p - d * (k - 1) - 1) // s + 1

        if dim == 0:
            ho = v
        else:
            wo = v

    return n, co, ho, wo


class MnistConvolution(nn.Module):
    def __init__(self, input_size=28,hidden_size=64, conv_num=32, kernel_size=2, explicit_permute=False):
        super(MnistConvolution, self).__init__()

        # self.hidden_size_sqrt = int(math.sqrt(hidden_size))
        # self.hidden_size = int(self.hidden_size_sqrt ** 2)
        # self.input_layer = nn.Linear(784, self.hidden_size)

        self.input_size = input_size
        self.conv_num = conv_num
        self.kernel_size = kernel_size

        self.padding = 1
        self.dilation = 1
        self.stride = 1
        self.explicit_permute = explicit_permute

        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=conv_num, kernel_size=kernel_size,
                                    stride=self.stride, padding=self.padding, dilation=self.dilation)

        size = conv2d_output_size(self.conv_layer, (1, self.input_size, self.input_size))
        self.conv_output_size = size[1] * size[2] * size[3]

        self.output_layer = nn.Linear(self.conv_output_size, 10)

    def forward(self, x):
        if self.explicit_permute:
            x = x.permute(0, 3, 1, 2)

        x = self.conv_layer(x)

        x = x.view(-1, self.conv_output_size)

        x = F.relu(self.output_layer(x))
        x = F.softmax(x, dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= 8
        return num_features


def main():
    import sys
    from MixedPrecision.pytorch.mnist_fully_connected import load_mnist
    from MixedPrecision.pytorch.mnist_fully_connected import train
    from MixedPrecision.pytorch.mnist_fully_connected import init_weights
    from MixedPrecision.tools.args import get_parser
    from MixedPrecision.tools.utils import summary
    import MixedPrecision.tools.utils as utils

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    parser = get_parser()
    args = parser.parse_args()

    utils.set_use_gpu(args.gpu)
    utils.set_use_half(args.half)

    input_size = 28
    if args.fake256:
        input_size = 256

    for k, v in vars(args).items():
        print('{:>30}: {}'.format(k, v))

    try:
        current_device = torch.cuda.current_device()
        print('{:>30}: {}'.format('GPU Count', torch.cuda.device_count()))
        print('{:>30}: {}'.format('GPU Name', torch.cuda.get_device_name(current_device)))
    except:
        pass

    model = MnistConvolution(
        input_size=input_size,
        hidden_size=args.hidden_size,
        conv_num=args.conv_num,
        kernel_size=args.kernel_size,
        explicit_permute=args.permute)

    model.float()
    model.apply(init_weights)
    model = utils.enable_cuda(model)
    summary(model, input_size=(args.batch_size, 784, 1))
    model = utils.enable_half(model)

    train(args, model, load_mnist(args, hwc_permute=args.permute, fake_256=args.fake256))

    sys.exit(0)


if __name__ == '__main__':
    main()

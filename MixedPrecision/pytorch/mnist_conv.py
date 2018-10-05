#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MnistConvolution(nn.Module):
    def __init__(self, hidden_size=64, conv_num=32, kernel_size=2):
        super(MnistConvolution, self).__init__()

        self.hidden_size_sqrt = int(math.sqrt(hidden_size))
        self.hidden_size = int(self.hidden_size_sqrt ** 2)
        self.conv_num = conv_num
        self.kernel_size = kernel_size

        self.padding = 1
        self.dilation = 1
        self.stride = 1

        self.input_layer = nn.Linear(784, self.hidden_size)
        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=conv_num, kernel_size=kernel_size,
                                    stride=self.stride, padding=self.padding, dilation=self.dilation)

        self.conv_output_size_base = ((self.hidden_size_sqrt + 2 * self.padding - self.dilation * (
                    kernel_size - 1) - 1) // self.stride + 1)
        self.conv_output_size = conv_num * self.conv_output_size_base * self.conv_output_size_base

        self.output_layer = nn.Linear(self.conv_output_size, 10)

    def forward(self, x):
        # Resize the input layer to fit our convolution
        x = x.view(-1, 784)
        x = F.relu(self.input_layer(x))

        # re shape the tensor for our convolution
        x = x.view(-1, 1, self.hidden_size_sqrt, self.hidden_size_sqrt)

        x = self.conv_layer(x)

        x = x.view(-1, self.conv_output_size)

        x = F.relu(self.output_layer(x))
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

    for k, v in vars(args).items():
        print('{:>30}: {}'.format(k, v))

    try:
        current_device = torch.cuda.current_device()
        print('{:>30}: {}'.format('GPU Count', torch.cuda.device_count()))
        print('{:>30}: {}'.format('GPU Name', torch.cuda.get_device_name(current_device)))
    except:
        pass

    model = MnistConvolution(
        hidden_size=args.hidden_size,
        conv_num=args.conv_num,
        kernel_size=args.kernel_size)

    if not utils.use_half():
        model.apply(init_weights)

    model = utils.enable_cuda(model)
    model = utils.enable_half(model)

    summary(model, input_size=(args.batch_size, 1, 784))

    train(args, model, load_mnist(args))

    sys.exit(0)


if __name__ == '__main__':
    main()

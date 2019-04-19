import torch
import torch.nn.functional as F
import torch.nn as nn

from MixedPrecision.pytorch.models.regressors import KernelRegressor


def simple_conv2d(m, k, **kwargs):
    """  m:        inC x  H x  W
         k: outC x inC x kH x kW """

    x = F.conv2d(m.view(1, *m.shape), k, **kwargs)
    _, c, h, w = x.shape
    return x.view(c, h, w)


def test_simple_conv2d():
    a = torch.rand(3, 10, 10)
    k = torch.rand(1, 3, 2, 2)
    assert simple_conv2d(a, k).shape == (1, 9, 9)

    a = torch.rand(3, 10, 10)
    k = torch.rand(12, 3, 2, 2)
    assert simple_conv2d(a, k).shape == (12, 9, 9)


def conv2d_iwk(images, kernels, **kwargs):
    """ Apply N kernels to a batch of N images

        Images : N x inC x H x W
        Kernels: N x outC x inC x kH x kW
        Output : N x outC x size(Conv2d)
    """

    data = []
    for image, out_kernels in zip(images, kernels):
        val = simple_conv2d(image, out_kernels, **kwargs)

        c, h, w = val.shape

        data.append(val.view(1, c, h, w))

    return torch.cat(data)


def test_conv2d_iwk():
    out_channel = 14
    in_channel = 3
    batch_size = 4

    imgs = torch.rand(batch_size, in_channel, 10, 10)
    ks = torch.rand(batch_size, out_channel, in_channel, 2, 2)

    assert conv2d_iwk(imgs, ks).shape == (batch_size, out_channel, 9, 9)


class HOConv2d(nn.Module):
    """
        Higher order Conv2d, the convolution has no parameters, the kernel are computed by a neural net
    """
    def __init__(self, input_shape, out_channel, kernel_size=(3, 2, 2), **kwargs):
        super(HOConv2d, self).__init__()
        self.regressor = KernelRegressor(input_shape, out_channel, kernel_size)
        self.kwargs = kwargs

    def forward(self, input):
        kernels = self.regressor(input)
        return conv2d_iwk(input, kernels, **self.kwargs)


if __name__ == '__main__':
    test_simple_conv2d()
    test_conv2d_iwk()

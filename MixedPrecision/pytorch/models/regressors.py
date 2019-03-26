import torch
import torch.nn as nn
import torch.utils.data

import torch.nn.functional as F
from functools import reduce


class ConvRegressor(torch.nn.Module):
    """ Given a batch of images returns the convolution
        kernel that should applied to each one of them

        input_shape = (c, h, w)
        output_shape: (Cout, Cin, h, w)
    """

    def __init__(self, input_shape, output_shape=(1, 3, 2, 2)):
        super(ConvRegressor, self).__init__()
        self.output_shape = output_shape

        in_channel, h, w = input_shape

        # Given a batch of images return a feature set
        # from which the kernel will be computed
        self.features = nn.Sequential(
            nn.Conv2d(in_channel, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # get the output size
        _, c, h, w = self.features(torch.rand(1, *input_shape)).shape
        self.feature_size = c * h * w
        self.output_size = reduce(lambda x, y: x * y, self.output_shape)

        # Given the previous features for a given batch returns
        # the conv kernel that should be applied
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size // 4),
            nn.ReLU(True),
            nn.Linear(self.feature_size // 4, self.output_size)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.feature_size)
        x = self.regressor(x)
        x = x.view(-1, *self.output_shape)
        return x


class TransformRegressor(ConvRegressor):
    """ This network regress the transformation matrix for a given image """

    def __init__(self, input_shape):
        super(TransformRegressor, self).__init__(input_shape, output_shape=(1, 2, 3))
        # Initialize the weights/bias with identity transformation
        self.regressor[2].weight.data.zero_()
        self.regressor[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def remove_transform(self, x):
        """ remove the transformation form an image X """
        theta = self.forward(x)
        grid = F.affine_grid(theta, x.size())
        return F.grid_sample(x, grid)


class KernelRegressor(ConvRegressor):
    """ Given a batch of images returns the convolution
        kernel that should applied to each one of them

        input_shape = (c, h, w)
        out_channel: int
        kernel_size: (c, h, w)
    """

    def __init__(self, input_shape, out_channel, kernel_size=(3, 2, 2)):
        super(KernelRegressor, self).__init__(input_shape, output_shape=(out_channel, *kernel_size))


def test_KernelFinder():
    out_channel = 14
    in_channel = 3
    batch_size = 4
    kernel_size = (1, 2, 2)
    img_size = 32

    model = KernelRegressor(
        input_shape=(in_channel, img_size, img_size),
        out_channel=out_channel,
        kernel_size=kernel_size)

    imgs = torch.rand(batch_size, in_channel, img_size, img_size)

    assert model(imgs).shape == (batch_size, out_channel, *kernel_size)


if __name__ == '__main__':
    test_KernelFinder()

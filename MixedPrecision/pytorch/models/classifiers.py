import torch
import torch.nn as nn
import torch.nn.functional as F

from MixedPrecision.pytorch.models.regressors import TransformRegressor
from MixedPrecision.pytorch.models.convs import HOConv2d


class ConvClassifier(nn.Module):
    def __init__(self, input_shape=(1, 28, 28)):
        super(ConvClassifier, self).__init__()

        c, h, w = input_shape

        self.convs = nn.Sequential(
            nn.Conv2d(c, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2)
        )

        _, c, h, w = self.convs(torch.rand(1, *input_shape)).shape
        self.conv_output_size = c * h * w

        self.fc1 = nn.Linear(self.conv_output_size, self.conv_output_size // 4)
        self.fc2 = nn.Linear(self.conv_output_size // 4, 10)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.conv_output_size)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class SpatialTransformerClassifier(nn.Module):
    """ Apply a classifier algorithm after removing an affine transform """

    def __init__(self, input_shape=(3, 32, 32)):
        super(SpatialTransformerClassifier, self).__init__()
        self.untransform = TransformRegressor(input_shape)
        self.classifier = ConvClassifier(input_shape)

    def forward(self, x):
        x = self.untransform.remove_transform(x)
        return self.classifier(x)


class HOConvClassifier(nn.Module):
    def __init__(self, input_shape=(1, 28, 28)):
        super(HOConvClassifier, self).__init__()

        c, h, w = input_shape

        self.conv1 = nn.Sequential(
            HOConv2d(input_shape=input_shape, out_channel=10, kernel_size=(c, 5, 5)),
            nn.MaxPool2d(2),
            nn.ReLU(True)
        )

        o1 = self.conv1(torch.rand(1, *input_shape))
        _, c, h, w = o1.shape

        self.conv2 = nn.Sequential(
            HOConv2d(input_shape=(c, h, w), out_channel=20, kernel_size=(c, 5, 5)),
            nn.Dropout2d(),
            nn.MaxPool2d(2)
        )

        _, c, h, w = self.conv2(o1).shape
        self.conv_output_size = c * h * w

        self.fc1 = nn.Linear(self.conv_output_size, self.conv_output_size // 4)
        self.fc2 = nn.Linear(self.conv_output_size // 4, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, self.conv_output_size)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':

    import sys
    sys.stderr = sys.stdout

    mnist_cls = ConvClassifier(input_shape=(1, 28, 28))
    mnist_cls(torch.rand(4, 1, 28, 28))

    imagenet_cls = ConvClassifier(input_shape=(3, 224, 224))
    imagenet_cls(torch.rand(4, 3, 224, 224))

    imagenet_hocls = HOConvClassifier(input_shape=(3, 224, 224))
    imagenet_hocls(torch.rand(4, 3, 224, 224))

    mnist_hocls = HOConvClassifier(input_shape=(1, 28, 28))
    mnist_hocls(torch.rand(4, 1, 28, 28))

from TruncatedResNet import TruncatedResNet
import torch
import torch.nn as nn
import torchvision.models.resnet as resnet


class HybridLoss:
    def __init__(self, mode=0.1):
        self.classifier_loss_function = nn.CrossEntropyLoss()
        self.regression_loss_function = nn.L1Loss(reduction='elementwise_mean')
        self.mode = mode
        self.classifier_loss = 0
        self.regression_loss = 0

    def __call__(self, y, y_hat, x, x_hat):
        # x_hat has only one channel

        with torch.no_grad():
            x_gray = x.mean(dim=1)

        # return self.mode * self.classifier_loss(y_hat, y) + (1 - self.mode) * ((x_hat - x_gray) ** 2)
        self.classifier_loss = self.classifier_loss_function(y_hat, y)
        self.regression_loss = self.regression_loss_function(x_hat, x_gray) ** 0.25  #((x_hat - x_gray) ** 2).mean()

        v = self.classifier_loss * self.regression_loss

        self.classifier_loss = self.classifier_loss.item()
        self.regression_loss = self.regression_loss.item()
        return v

    def cuda(self):
        self.classifier_loss_function.cuda()
        return self


class HybridClassifier(nn.Module):

    def __init__(self, classes=1000, output=(224, 224)):
        super(HybridClassifier, self).__init__()
        # self.resizer = nn.Linear(3 * 32 * 32, 3 * 224 * 224)
        # torch.nn.ConvTranspose2d()

        self.rectify_fc2_backward = []
        self.classify_fc_backward = []

        self.resnet = TruncatedResNet(resnet.BasicBlock, [2, 2, 2, 2])
        self.classify_fc = nn.Linear(512 * resnet.BasicBlock.expansion, classes)
        # self.classify_fc.register_backward_hook(lambda x: self.classify_fc_backward.append(x.mean().item()))

        # torch.Size([64, 512, 7, 7]) => torch.Size([64, 8, 15, 15])
        self.rectify_fc1 = nn.ConvTranspose2d(512, 8, kernel_size=3, stride=2)
        # torch.Size([64, 8, 15, 15]) => 64, 1, 224, 224
        self.rectify_fc2 = nn.Linear(8 * 15 * 15, output[0] * output[1])
        # self.rectify_fc2.register_backward_hook(lambda x: self.rectify_fc2_backward.append(x.mean().item()))

        self.output = output
        self.name = 'Experience1b'

    def forward(self, x):
        # x = self.resizer(x.view(-1, 3, 32, 32))
        out, last = self.resnet(x)
        y_hat = self.classify_fc(out)

        # print(last.shape) = torch.Size([64, 512, 7, 7]) => 25088
        # target => 224x224 => 50176
        x_hat = self.rectify_fc1(last)
        # print(x_hat.shape) => torch.Size([64, 3, 15, 15])

        # x_hat = self.rectify_fc2(x_hat)
        x_hat = self.rectify_fc2(x_hat.view(-1, 8 * 15 * 15))

        return y_hat, x_hat.view(-1, self.output[0], self.output[1])

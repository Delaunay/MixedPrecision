#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torchvision import datasets

from MixedPrecision.tools.fakeit import fakeit


class MnistFullyConnected(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), hidden_size=64, hidden_num=0):
        super(MnistFullyConnected, self).__init__()
        self.input_shape = input_shape
        self.hidden_num = hidden_num
        self.hidden_size = hidden_size

        self.input_layer = nn.Linear(self.input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(0, hidden_num)])
        self.output_layer = nn.Linear(hidden_size, 10)

    @property
    def input_size(self):
        return self.input_shape[0] * self.input_shape[1] * self.input_shape[2]

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.relu(self.input_layer(x))

        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))

        x = F.softmax(self.output_layer(x), dim=1)
        return x


def load_mnist(args, fake_data=False, hwc_permute=False, shape=(1, 28, 28)):
    import MixedPrecision.tools.utils as utils

    perm = transforms.Lambda(lambda x: x)

    if hwc_permute:
        perm = transforms.Lambda(lambda x: x.permute(1, 2, 0))

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        perm,
        transforms.Lambda(lambda x: utils.enable_half(utils.enable_cuda(x)))
    ])

    dataset = None

    if fake_data:
        dataset = fakeit('pytorch', args.batch_size, shape, 10, trans)
    else:
        dataset = datasets.MNIST(args.data + '/', train=True, download=True, transform=trans)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    return train_loader


def train(args, model, data):
    import time

    import MixedPrecision.tools.utils as utils
    from MixedPrecision.tools.optimizer import OptimizerAdapter
    from MixedPrecision.tools.stats import StatStream

    model = utils.enable_cuda(model)
    model = utils.enable_half(model)

    criterion = utils.enable_cuda(nn.CrossEntropyLoss())
    criterion = utils.enable_half(criterion)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    optimizer = OptimizerAdapter(
        optimizer,
        static_loss_scale=args.static_loss_scale,
        dynamic_loss_scale=args.dynamic_loss_scale
    )

    model.train()

    compute_time = StatStream(1)
    floss = float('inf')

    for epoch in range(0, args.epochs):
        cstart = time.time()

        for batch in data:
            x, y = batch

            x = utils.enable_cuda(x)
            y = utils.enable_cuda(y)

            x = utils.enable_half(x)

            out = model(x)
            loss = criterion(out, y)

            floss = loss.item()

            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()

        cend = time.time()
        compute_time += cend - cstart

        print('[{:4d}] Compute Time (avg: {:.4f}, sd: {:.4f}) Loss: {:.4f}'.format(
            1 + epoch, compute_time.avg, compute_time.sd, floss))


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0.01)


def main():
    import sys
    from MixedPrecision.tools.args import get_parser
    from MixedPrecision.tools.utils import summary
    import MixedPrecision.tools.utils as utils

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    parser = get_parser()
    args = parser.parse_args()

    utils.set_use_gpu(args.gpu)
    utils.set_use_half(args.half)

    shape = (1, 28, 28)
    if args.fake:
        shape = args.shape

    for k, v in vars(args).items():
        print('{:>30}: {}'.format(k, v))

    try:
        current_device = torch.cuda.current_device()
        print('{:>30}: {}'.format('GPU Count', torch.cuda.device_count()))
        print('{:>30}: {}'.format('GPU Name', torch.cuda.get_device_name(current_device)))
    except:
        pass

    model = MnistFullyConnected(input_shape=shape, hidden_size=args.hidden_size, hidden_num=args.hidden_num)
    model.float()
    model.apply(init_weights)
    model = utils.enable_cuda(model)
    summary(model, input_size=(args.batch_size, 1, 28, 28))
    model = utils.enable_half(model)

    train(args, model, load_mnist(args, hwc_permute=args.permute, fake_data=args.fake, shape=shape))

    sys.exit(0)


if __name__ == '__main__':
    main()

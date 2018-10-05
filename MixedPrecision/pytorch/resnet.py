import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel

import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from MixedPrecision.pytorch.ResNet import resnet18

import MixedPrecision.tools.utils as utils
from PIL import Image



def load_imagenet(args):
    import torchvision.transforms as transforms

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    img = Image.open(args.data + '/train/n01440764/n01440764_8082.JPEG')
    data = utils.enable_cuda(torch.stack([utils.enable_half(transforms(img)) for i in range(0, args.batch_size)]))

    target = utils.enable_cuda(torch.tensor([i for i in range(0, args.batch_size)])).long()
    target = target[torch.randperm(args.batch_size)]
    return data, target


def train(args, model, data):
    import time

    import MixedPrecision.tools.utils as utils
    from MixedPrecision.tools.optimizer import OptimizerAdapter
    from MixedPrecision.tools.stats import StatStream

    model = utils.enable_cuda(model)
    model = utils.enable_half(model)

    criterion = utils.enable_cuda(nn.CrossEntropyLoss())
    criterion = utils.enable_half(criterion)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    optimizer = OptimizerAdapter(
        optimizer,
        half=args.half,
        static_loss_scale=args.static_loss_scale,
        dynamic_loss_scale=args.dynamic_loss_scale
    )
    model.train()

    compute_time = StatStream(drop_first_obs=1)
    epoch_size = 100
    floss = float('inf')

    x, y = data

    for epoch in range(0, args.epochs):
        compute_start = time.time()

        for i in range(0, epoch_size):
            # measure data loading time
            x = utils.enable_cuda(x)
            y = utils.enable_cuda(y)

            x = utils.enable_half(x)
            y = utils.enable_half(y)

            # compute output
            output = model(x)
            loss = criterion(output, y)
            floss = loss.item()

            # compute gradient and do SGD step
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()

        compute_end = time.time()
        compute_time.update(compute_end - compute_start)

        print('[{:4d}] Compute Time (avg: {:.4f}, sd: {:.4f}) Loss: {:.4f}'.format(
            1 + epoch, compute_time.avg, compute_time.sd, floss))

def main():
    import sys
    from MixedPrecision.tools.args import get_parser
    from MixedPrecision.tools.utils import summary

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    parser = get_parser()
    args = parser.parse_args()

    utils.set_use_gpu(args.gpu)
    utils.set_use_half(args.half)

    for k, v in vars(args).items():
        print('{:>30}: {}'.format(k, v))

    current_device = torch.cuda.current_device()
    print('{:>30}: {}'.format('GPU Count', torch.cuda.device_count()))
    print('{:>30}: {}'.format('GPU Name', torch.cuda.get_device_name(current_device)))

    model = utils.enable_cuda(resnet18(_half=args.half))
    model = utils.enable_half(model)

    summary(model, input_size=(3, 224, 224))

    train(args, model, load_imagenet(args))

    sys.exit(0)


if __name__ == '__main__':
    main()

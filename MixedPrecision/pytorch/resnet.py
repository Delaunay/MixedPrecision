import torch
import torch.nn as nn
import torch.nn.parallel

import torch.optim

import torch.utils.data
import torch.utils.data.distributed

import MixedPrecision.tools.utils as utils

import torchvision.models.resnet as resnet
import torchvision.transforms as transforms
import torchvision.datasets as datasets

"""
def load_imagenet(args):
    import torchvision.transforms as transforms
    from PIL import Image

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
    data = utils.enable_half(utils.enable_cuda(data))

    target = utils.enable_cuda(torch.tensor([i for i in range(0, args.batch_size)])).long()
    target = target[torch.randperm(args.batch_size)]
    target = utils.enable_half(utils.enable_cuda(target))
    return data, target*/"""

data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip()
    # transforms.ToTensor(),
    # normalize,
    # transforms.Lambda(lambda x: utils.enable_cuda(utils.enable_half(x)))
])


def load_imagenet(args):
    global data_transforms

    print('Loading imagenet from {}'.format(args.data))

    train_dataset = datasets.ImageFolder(
        args.data + '/train/',
        data_transforms)

    return torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=None,
        num_workers=args.workers, pin_memory=True, collate_fn=utils.fast_collate)


def fake_imagenet(args):
    from MixedPrecision.tools.fakeit import fakeit
    global data_transforms

    print('Faking Imagenet data')
    target_transform = transforms.Compose([
        transforms.Lambda(lambda x: utils.enable_cuda(x.long()))
    ])

    dataset = fakeit('pytorch', args.batch_size * 10, (3, 224, 244), 1000, data_transforms, target_transform)

    return torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, pin_memory=True,
        num_workers=args.workers, shuffle=None, collate_fn=utils.fast_collate
    )


def train(args, model, dataset):
    import time

    import MixedPrecision.tools.utils as utils
    from MixedPrecision.tools.optimizer import OptimizerAdapter
    from MixedPrecision.tools.stats import StatStream
    from MixedPrecision.tools.prefetcher import DataPreFetcher

    model = utils.enable_cuda(model)
    model = utils.enable_half(model)

    criterion = utils.enable_cuda(nn.CrossEntropyLoss())
    # No Half precision for the criterion
    # criterion = utils.enable_half(criterion)

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

    epoch_compute = StatStream(drop_first_obs=1)
    batch_compute = StatStream(drop_first_obs=10)
    floss = float('inf')

    mean = utils.enable_half(torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).float()).view(1, 3, 1, 1)
    std = utils.enable_half(torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).float()).view(1, 3, 1, 1)

    for epoch in range(0, args.epochs):
        epoch_compute_start = time.time()

        data = DataPreFetcher(dataset, mean=mean, std=std)
        x, y = data.next()

        while x is not None:
            # compute output
            batch_compute_start = time.time()
            output = model(x)
            loss = criterion(output, y)
            floss = loss.item()

            # compute gradient and do SGD step
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()

            torch.cuda.synchronize()
            batch_compute_end = time.time()
            batch_compute += batch_compute_end - batch_compute_start

            x, y = data.next()

        epoch_compute_end = time.time()
        epoch_compute.update(epoch_compute_end - epoch_compute_start)

        print('[{:4d}] Epoch Time (avg: {:.4f}, sd: {:.4f}) Batch Time (avg: {:.4f}, sd: {:.4f}) Loss: {:.4f}'.format(
            1 + epoch, epoch_compute.avg, epoch_compute.sd, batch_compute.avg, batch_compute.sd, floss))


def generic_main(make_model):
    import sys
    from MixedPrecision.tools.args import get_parser
    from MixedPrecision.tools.utils import summary
    from apex.fp16_utils import network_to_half

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

    model = utils.enable_cuda(make_model())

    if utils.use_half():
        model = network_to_half(model)
    else:
        model = utils.enable_half(model)

    summary(model, input_size=(3, 224, 224))

    data = None
    if args.fake:
        data = fake_imagenet(args)
    else:
        data = load_imagenet(args)

    train(args, model, data)

    sys.exit(0)


def resnet18_main():
    return generic_main(resnet.resnet18)


def resnet50_main():
    return generic_main(resnet.resnet50)


if __name__ == '__main__':
    resnet50_main()

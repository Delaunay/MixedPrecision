import torch
import torch.nn.parallel

import torch.utils.data
import torch.utils.data.distributed

import torchvision
import torchvision.transforms as transforms


def default_pytorch_loader(args):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        args.data,
        data_transforms)

    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=None,
        num_workers=args.workers,
        pin_memory=True
    )


def prefetch_pytorch_loader(args):
    from MixedPrecision.tools.prefetcher import DataPreFetcher
    from MixedPrecision.tools.stats import StatStream
    import MixedPrecision.tools.utils as utils

    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        args.data,
        data_transforms)

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=None,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=utils.timed_fast_collate
    )

    mean = utils.enable_half(torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).float()).view(1, 3, 1, 1)
    std = utils.enable_half(torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).float()).view(1, 3, 1, 1)

    return DataPreFetcher(
        loader,
        mean=mean, std=std,
        cpu_stats=StatStream(drop_first_obs=10),
        gpu_stats=StatStream(drop_first_obs=10)
    )


def dali_loader(args):
    from MixedPrecision.tools.dali import make_dali_loader

    return make_dali_loader(
        args,
        args.data,
        224
    )


def benzina_loader(args):
    import MixedPrecision.tools.benzina as benzina

    return benzina.make_data_loader(args, 224)


def ziparchive_loader(args):
    from MixedPrecision.tools.prefetcher import DataPreFetcher
    from MixedPrecision.tools.stats import StatStream
    from MixedPrecision.tools.archive import ZipDataset
    import MixedPrecision.tools.utils as utils

    dataset = ZipDataset(
        args.data,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])
     )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.workers,
        shuffle=None,
        collate_fn=utils.timed_fast_collate
    )

    mean = utils.enable_half(torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).float()).view(1, 3, 1, 1)
    std = utils.enable_half(torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).float()).view(1, 3, 1, 1)

    return DataPreFetcher(
        loader,
        mean=mean, std=std,
        cpu_stats=StatStream(drop_first_obs=10),
        gpu_stats=StatStream(drop_first_obs=10)
    )


def fake_imagenet(args):
    from MixedPrecision.tools.fakeit import fakeit
    global data_transforms

    print('Faking Imagenet data')
    target_transform = transforms.Compose([
        transforms.Lambda(lambda x: utils.enable_cuda(x.long()))
    ])

    dataset = fakeit('pytorch', args.batch_size * 10, (3, 224, 224), 1000, data_transforms, target_transform)

    return torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, pin_memory=True,
        num_workers=args.workers, shuffle=None, collate_fn=utils.timed_fast_collate
    )


def benchmark_loader(args):
    import time
    import socket

    from MixedPrecision.tools.stats import StatStream
    import MixedPrecision.tools.report as report

    loader = {
        'torch': default_pytorch_loader,
        'prefetch': prefetch_pytorch_loader,
        'benzina': benzina_loader,
        'dali': dali_loader,
        'zip': ziparchive_loader
    }

    data = loader[args.loader](args)
    stat = StatStream(10)
    prof = args.prof

    for i in range(0, args.epochs):
        start = time.time()
        for j, (x, y) in enumerate(data):
            if j > prof:
                break

        end = time.time()
        stat += end - start

    hostname = socket.gethostname()
    current_device = torch.cuda.current_device()
    gpu = torch.cuda.get_device_name(current_device)
    bs = args.batch_size

    common = [args.batch_size, args.workers, args.loader, hostname, gpu]
    report.print_table(
        ['Metric', 'Average', 'Deviation', 'Min', 'Max', 'count', 'batch', 'workers', 'loader', 'hostname', 'GPU'],
        [
            ['Load Time (s)'] + stat.to_array() + common,
            ['Load Speed (img/s)', bs * prof / stat.avg, 'NA', bs * prof / stat.max, bs * prof / stat.min, stat.count]
        ]
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Data loader Benchmark')

    parser.add_argument('--data', type=str, metavar='DIR',
                        help='path to the dataset location')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')

    parser.add_argument('--prof', dest='prof', type=int, default=10,
                        help='Only run N iterations for profiling.')

    parser.add_argument('--loader', type=str, default='pytorch',
                        help='The kind of loader to use (torch, prefetch, benzina, dali, zip)')

    args = parser.parse_args()
    benchmark_loader(args)


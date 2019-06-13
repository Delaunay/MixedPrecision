import torch
import torch.nn.parallel

import torch.utils.data
import torch.utils.data.distributed

import torchvision
import torchvision.transforms as transforms

from MixedPrecision.tools.dataloader import TimedImageFolder


def default_cifar10_loader(args, train=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=args.data,
        train=train,
        download=True,
        transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader


def default_cifar100_loader(args, train=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
    ])

    trainset = torchvision.datasets.CIFAR100(
        root=args.data,
        train=train,
        download=True,
        transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader


def default_pytorch_loader(args, train=True):
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

    train_dataset = TimedImageFolder(
        args.data,
        data_transforms)

    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)


def prefetch_pytorch_loader(args, train=True, pin_memory=True):
    from MixedPrecision.tools.prefetcher import DataPreFetcher
    from MixedPrecision.tools.stats import StatStream
    import MixedPrecision.tools.utils as utils

    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ])

    train_dataset = TimedImageFolder(
        args.data,
        data_transforms)

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
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


def dali_loader(args, train=True):
    from MixedPrecision.tools.dali import make_dali_loader

    return make_dali_loader(
        args,
        args.data,
        224
    )


def benzina_loader(args, train=True):
    import MixedPrecision.tools.benzina as benzina

    return benzina.make_data_loader(args, 224)


def hdf5_loader(args, train=True):
    from MixedPrecision.tools.hdf5 import HDF5Dataset

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    data_transforms = transforms.Compose([
        # data is stored as uint8
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = HDF5Dataset(
        args.data,
        data_transforms)

    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)


def ziparchive_loader(args, train=True):
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
        shuffle=True,
        collate_fn=utils.timed_fast_collate
    )

    mean = utils.enable_half(torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).float()).view(1, 3, 1, 1)
    std = utils.enable_half(torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).float()).view(1, 3, 1, 1)

    return DataPreFetcher(
        loader,
        mean=mean, std=std,
        cpu_stats=StatStream(drop_first_obs=10),
    )


def fake_imagenet(args, train=True):
    from MixedPrecision.tools.fakeit import fakeit
    import MixedPrecision.tools.utils as utils

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

    print('Faking Imagenet data')
    dataset = fakeit('pytorch', args.batch_size * 10, (3, 224, 224), 1000, data_transforms)

    return torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, pin_memory=True,
        num_workers=args.workers, shuffle=None, collate_fn=utils.timed_fast_collate
    )


def load_dataset(args, train=True):
    from MixedPrecision.tools.prefetcher import AsyncPrefetcher

    loader = {
        'torch': default_pytorch_loader,
        'prefetch': prefetch_pytorch_loader,
        'benzina': benzina_loader,
        'dali': dali_loader,
        'zip': ziparchive_loader,
        'hdf5': hdf5_loader,
        'fake': fake_imagenet,
        'torch_cifar10': default_cifar10_loader,
        'torch_cifar100': default_cifar100_loader,
    }

    data = loader[args.loader](args, train)

    #if args.async:
    #    data = AsyncPrefetcher(data, buffering=2)

    return data


def benchmark_loader(args):
    import time
    import socket

    from MixedPrecision.tools.stats import StatStream
    from MixedPrecision.tools.prefetcher import AsyncPrefetcher
    import MixedPrecision.tools.report as report

    def ignore(x, y):
        pass

    s = time.time()

    data = load_dataset(args)

    stat = StatStream(20)
    prof = args.prof
    print('Init time was {:.4f}'.format(time.time() - s))
    print('Starting..')

    start = time.time()

    for j, (x, y) in enumerate(data):
        #x = x.cuda()
        #y = y.cuda()

        ignore(x, y)

        end = time.time()
        current_time = end - start
        stat += current_time

        if j > prof:
            break

        if stat.avg > 0:
            print('[{:4d}] {:.4f} (avg: {:.4f} img/s)'.format(j, args.batch_size / current_time, args.batch_size / stat.avg))

        start = time.time()

    print('Done')

    hostname = socket.gethostname()
    #current_device = torch.cuda.current_device()
    #gpu = torch.cuda.get_device_name(current_device)
    gpu = 0
    bs = args.batch_size

    common = [args.batch_size, args.workers, args.loader, hostname, gpu]
    report.print_table(
        ['Metric', 'Average', 'Deviation', 'Min', 'Max', 'count', 'batch', 'workers', 'loader', 'hostname', 'GPU'],
        [
            ['Load Time (s)'] + stat.to_array() + common,
            ['Load Speed (img/s)', bs / stat.avg, 'NA', bs / stat.max, bs / stat.min, stat.count] + common
        ]
    )


def main():
    # This does not work but this is what the documentation says to do...
    #try:
    #    import torch.multiprocessing as multiprocessing
    #    multiprocessing.set_start_method('spawn')
    #except Exception as e:
    #    print(e)

    import MixedPrecision.tools.utils as utils
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

    parser.add_argument('--async', action='store_true', default=False,
                        help='Use AsyncPrefetcher')

    args = parser.parse_args()

    utils.set_use_gpu(True, True)
    utils.set_use_half(True)

    utils.show_args(args)

    benchmark_loader(args)


if __name__ == '__main__':
    main()

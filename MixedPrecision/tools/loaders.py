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


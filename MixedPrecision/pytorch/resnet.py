import torch
import torch.nn as nn
import torch.nn.parallel

import torch.optim

import torch.utils.data
import torch.utils.data.distributed

import torchvision
import torchvision.models.resnet as resnet
import torchvision.transforms as transforms

import MixedPrecision.tools.utils as utils
import MixedPrecision.tools.report as report

import MixedPrecision.tools.dataloader as datasets
import MixedPrecision.tools.benzina as benzina
from MixedPrecision.tools.fakeit import TarDataset

import math
import socket
import psutil
import os

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

    if args.accimage:
        import accimage
        torchvision.set_image_backend('accimage')

    print('Using `{}` as image loader'.format(torchvision.get_image_backend()))

    print('Loading imagenet from {}'.format(args.data))
    if args.use_dali:
        print('Using Dali dataloader')
        from MixedPrecision.tools.dali import make_dali_loader

        return make_dali_loader(
            args,
            args.data,
            224
        )
    elif args.benzina:
        print('Using Benzina dataloader')
        return benzina.make_data_loader(args, 224)
    else:
        train_dataset = datasets.TimedImageFolder(
            args.data,
            data_transforms)

        return torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=None,
            num_workers=args.workers, pin_memory=True, collate_fn=utils.timed_fast_collate)


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


def current_stream():
    if utils.use_gpu():
        return torch.cuda.current_stream()
    return None


# TODO refactor this
def train(args, model, dataset, name, is_warmup=False):
    import time

    import MixedPrecision.tools.utils as utils
    from MixedPrecision.tools.optimizer import OptimizerAdapter
    from MixedPrecision.tools.stats import StatStream
    from MixedPrecision.tools.prefetcher import DataPreFetcher
    from MixedPrecision.tools.nvidia_smi import make_monitor
    from MixedPrecision.tools.prefetcher import AsyncPrefetcher

    from apex.fp16_utils import network_to_half

    model = utils.enable_cuda(model)

    if args.half:
        model = network_to_half(model)

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

    epoch_compute = StatStream(drop_first_obs=10)
    batch_compute = StatStream(drop_first_obs=10)
    gpu_compute = StatStream(drop_first_obs=10)
    compute_speed = StatStream(drop_first_obs=10)
    effective_speed = StatStream(drop_first_obs=10)
    data_waiting = StatStream(drop_first_obs=10)
    data_loading_gpu = StatStream(drop_first_obs=10)
    data_loading_cpu = StatStream(drop_first_obs=10)
    full_time = StatStream(drop_first_obs=10)
    iowait = StatStream(drop_first_obs=10)

    start_event = torch.cuda.Event(enable_timing=True, blocking=False, interprocess=False)
    end_event = torch.cuda.Event(enable_timing=True, blocking=False, interprocess=False)

    floss = float('inf')

    mean = utils.enable_half(torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).float()).view(1, 3, 1, 1)
    std = utils.enable_half(torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).float()).view(1, 3, 1, 1)

    # Stop after n print when benchmarking (n * batch_count) batch
    print_count = 0
    monitor_proc, gpu_monitor = make_monitor(loop_interval=250)

    def should_run():
        if args.prof is None:
            return True
        return print_count < args.prof

    try:
        for epoch in range(0, args.epochs):
            epoch_compute_start = time.time()

            # do not prefetch when using dali
            if args.use_dali:
                data = dataset
            else:
                data = DataPreFetcher(
                    dataset,
                    mean=mean, std=std,
                    cpu_stats=data_loading_cpu,
                    gpu_stats=data_loading_gpu
                )

            # data = AsyncPrefetcher(iter(dataset), 3)


            # Looks like it only compute for the current process and not the children
            #cpu_times_start = psutil.cpu_times()
            data_time_start = time.time()
            x, y = data.next()

            batch_count = 0
            effective_batch = 0

            while x is not None and should_run():
                data_time_end = time.time()
                cpu_times_end = psutil.cpu_times()
                data_waiting += (data_time_end - data_time_start)
                #iowait += cpu_times_end.iowait - cpu_times_start.iowait


                # compute output
                batch_compute_start = time.time()

                # Compute time using the GPU as well
                torch.cuda.current_stream().record_event(start_event)

                output = model(x)
                loss = criterion(output, y.long())
                floss = loss.item()

                # compute gradient and do SGD step
                optimizer.zero_grad()
                optimizer.backward(loss)
                optimizer.step()
                torch.cuda.current_stream().record_event(end_event)

                end_event.synchronize()
                gpu_compute += start_event.elapsed_time(end_event) / 1000.0

                batch_compute_end = time.time()
                full_time += batch_compute_end - data_time_start
                batch_compute += batch_compute_end - batch_compute_start

                compute_speed += args.batch_size / (batch_compute_end - batch_compute_start)
                effective_speed += args.batch_size / (batch_compute_end - data_time_start)

                effective_batch += 1

                #cpu_times_start = psutil.cpu_times()
                data_time_start = time.time()
                x, y = data.next()

                batch_count += 1

                if effective_batch % 10 == 0:

                    print_count += 1
                    speed_avg = args.batch_size / batch_compute.avg

                    print('[{:4d}][{:4d}] '
                          'Batch Time (avg: {batch_compute.avg:.4f}, sd: {batch_compute.sd:.4f}) ' 
                          'Speed (avg: {speed:.4f}) '
                          'Data (avg: {data_waiting.avg:.4f}, sd: {data_waiting.sd:.4f})'.format(
                            1 + epoch, batch_count, batch_compute=batch_compute, speed=speed_avg, data_waiting=data_waiting))

            epoch_compute_end = time.time()
            epoch_compute.update(epoch_compute_end - epoch_compute_start)

            if not should_run():
                monitor_proc.terminate()
                if not is_warmup:
                    hostname = socket.gethostname()
                    current_device = torch.cuda.current_device()
                    gpu = torch.cuda.get_device_name(current_device)

                    bs = args.batch_size
                    loader = 'pytorch'
                    if args.use_dali:
                        loader = 'dali'
                    if args.benzina:
                        loader = 'benzina'

                    header = ['Metric', 'Average', 'Deviation', 'Min', 'Max', 'count', 'half', 'batch', 'workers', 'loader', 'model', 'hostname', 'GPU', 'accimage']
                    common = [args.half, args.batch_size, args.workers, loader, name, hostname, gpu, args.accimage]

                    report_data = [
                        ['Waiting for data (s)'] + data_waiting.to_array() + common,
                        ['GPU Compute Time (s)'] + gpu_compute.to_array() + common,
                        ['Full Batch Time (s)'] + full_time.to_array() + common,
                        ['Compute Speed (img/s)', bs / batch_compute.avg, 'NA', bs / batch_compute.max, bs / batch_compute.min, batch_compute.count] + common,
                        ['Effective Speed (img/s)', bs / full_time.avg, 'NA', bs / full_time.max, bs / full_time.min, batch_compute.count] + common,
                        # Ignored Metric
                        #  GPU timed on the CPU side (very close to GPU timing anway)
                        # # ['CPU Compute Time (s)] + batch_compute.to_array() + common,

                        #  https://en.wikipedia.org/wiki/Harmonic_mean
                        # ['Compute Inst Speed (img/s)'] + compute_speed.to_array() + common,
                        # ['Effective Inst Speed (img/s)'] + effective_speed.to_array() + common,

                        # ['iowait'] + iowait.to_array() + common
                    ]

                    # Dali is just a black box..
                    # no metrics are available
                    if not (args.use_dali or args.benzina):
                        data_reading = dataset.dataset.read_timer
                        data_transform = dataset.dataset.transform_timer
                        collate_time = utils.timed_fast_collate.time_stream

                        report_data += [['Prefetch CPU Data loading (s)'] + data_loading_cpu.to_array() + common]
                        report_data += [['Prefetch GPU Data Loading (s)'] + data_loading_gpu.to_array() + common]
                        report_data += [['Read Time (s)'] + data_reading.to_array() + common]
                        report_data += [['Transform Time (s)'] + data_transform.to_array() + common]
                        report_data += [['Read Speed per process (img/s)', 1.0 / data_reading.avg, 'NA', 1.0 / data_reading.max, 1.0 / data_reading.min, data_reading.count] + common]
                        report_data += [['Transform Speed per process  (img/s)', 1.0 / data_transform.avg, 'NA', 1.0 / data_transform.max, 1.0 / data_transform.min, data_transform.count] + common]

                        report_data += [['Read Speed (img/s)', args.workers / data_reading.avg, 'NA', args.workers / data_reading.max, args.workers / data_reading.min, data_reading.count] + common]
                        report_data += [['Transform Speed (img/s)', args.workers / data_transform.avg, 'NA', args.workers / data_transform.max, args.workers / data_transform.min, data_transform.count] + common]
                        report_data += [['Image Aggregation Speed (img/s)', bs / collate_time.avg, 'NA', bs / collate_time.max, bs / collate_time.min, collate_time.count] + common]
                        report_data += [['Image Aggregation Time (s)', collate_time.avg, collate_time.sd, collate_time.max, collate_time.min, collate_time.count] + common]

                    #gpu_monitor.report()
                    report_data.extend(gpu_monitor.arrays(common))
                    report.print_table(header, report_data, filename=args.report)
                break
    finally:
        monitor_proc.terminate()


def generic_main(make_model, name):
    import sys
    from MixedPrecision.tools.args import get_parser
    from MixedPrecision.tools.utils import summary

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    parser = get_parser()
    args = parser.parse_args()

    utils.set_use_gpu(args.gpu, not args.no_bench_mode)
    utils.set_use_half(args.half)

    for k, v in vars(args).items():
        print('{:>30}: {}'.format(k, v))

    try:
        current_device = torch.cuda.current_device()
        print('{:>30}: {}'.format('GPU Count', torch.cuda.device_count()))
        print('{:>30}: {}'.format('GPU Name', torch.cuda.get_device_name(current_device)))
    except:
        pass

    model = make_model()

    summary(model, input_size=(3, 224, 224), batch_size=args.batch_size)

    data = None
    if args.fake:
        data = fake_imagenet(args)
    else:
        data = load_imagenet(args)

    if args.warmup:
        train(args, model, data, name, is_warmup=True)

    train(args, model, data, name, is_warmup=False)

    sys.exit(0)


def resnet18_main():
    return generic_main(resnet.resnet18, 'resnet18')


def resnet50_main():
    return generic_main(resnet.resnet50, 'resnet50')


if __name__ == '__main__':
    # print(mp.get_start_method())
    #mp.set_start_method('spawn')

    resnet18_main()

import torch
import torch.nn as nn
import torchvision.models as models
from benchutils.chrono import MultiStageChrono
import MixedPrecision.tools.loaders as loaders

# from apex import amp
import argparse


def main():
    # ----
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', metavar='DIR', help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR')
    parser.add_argument('--opt-level', type=str)
    parser.add_argument('--cuda', action='store_true', default=True,  dest='cuda')
    parser.add_argument('--no-cuda', action='store_false',  dest='cuda')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--loader', type=str, default='torch')
    parser.add_argument('--prof', type=int, default=None)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--sync-all', type=bool, default=False)

    args = parser.parse_args()
    chrono = MultiStageChrono(skip_obs=10, sync=None)

    device = torch.device('cpu')
    if torch.cuda.is_available() and args.cuda:
        device = torch.device('cuda')

    torch.set_num_threads(args.workers)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # --
    try:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
    except ImportError:
        pass

    # ----
    model = models.__dict__[args.arch]()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr)

    # # ----
    # model, optimizer = amp.initialize(
    #     model,
    #     optimizer,
    #     enabled=args.opt_level != 'O0',
    #     cast_model_type=None,
    #     patch_torch_functions=True,
    #     keep_batchnorm_fp32=None,
    #     master_weights=None,
    #     loss_scale="dynamic",
    #     opt_level=args.opt_level
    # )

    # ----
    train_loader = loaders.load_dataset(args, train=True)

    # dataset is reduced but should be big enough for benchmark!
    batch_iter = iter(train_loader)

    def next_batch(iterator):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(train_loader)
            return next(iterator), iterator

    batch_count = len(train_loader)
    if args.prof is not None:
        batch_count = args.prof

    sync_fun = lambda: torch.cuda.current_stream().synchronize()
    sub_syncs = None
    if args.sync_all:
        sub_syncs = sync_fun

    print('Computing...')
    model.train()
    for epoch in range(args.epochs):

        # we sync after batch_count to not slowdown things
        with chrono.time('train', skip_obs=1, sync=sync_fun) as timer:
            for _ in range(batch_count):

                # data loading do not start here so naturally this is not data loading
                # only the time waiting for the data loading to finish
                with chrono.time('loading', sync=sub_syncs):
                    (input, target), batch_iter = next_batch(batch_iter)

                    input = input.to(device)
                    target = target.to(device)

                # if we do not synchronize we only get cuda `launch time`
                # not the actual compute
                with chrono.time('compute', sync=sub_syncs):
                    output = model(input)
                    loss = criterion(output, target)

                    # compute gradient and do SGD step
                    optimizer.zero_grad()

                    # with amp.scale_loss(loss, optimizer) as scaled_loss:
                    #     scaled_loss.backward()

                    loss.backward()
                    optimizer.step()

        print(f'[{epoch:3d}/{args.epochs:3d}] ETA: {(args.epochs - epoch - 1) * timer.avg:6.2f} sec')

    print('--')
    print(chrono.to_json(indent=2))
    print('--')
    print(f'{(args.batch_size * batch_count) / chrono.chronos["train"].avg:6.2f} Img/sec')
    print('-' * 25)


if __name__ == '__main__':
    main()

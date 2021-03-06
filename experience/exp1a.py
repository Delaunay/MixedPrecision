from MixedPrecision.tools.args import get_parser
from MixedPrecision.tools.loaders import load_dataset
from baseline import Baseline

import torchvision.models.resnet as resnet


def embed_label(x, y):
    shape = x.shape
    size = shape[-1] * shape[-2]
    x = x.view(-1, 3, size)
    x[:, 0, y] = 0
    x[:, 1, y] = 0
    x[:, 2, y] = 0
    x = x.view(-1, 3, shape[-2], shape[-1])
    return x, y


class Experience1a(Baseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.name = self.name = 'Experience1a'
        self.load_model()

    def train_batch(self):
        self.chrono.start()
        index, (x, y) = self.fetch_input()

        x, y = embed_label(x, y)

        x = x.cuda()
        y = y.cuda()

        self.chrono.start()
        output = self.model(x)
        loss = self.criterion(output, y.long())

        acc = (output.max(dim=1)[1] == y).sum().item() / len(y)
        self.update_stat(acc, loss)

        self.chrono.start()
        self.backward(loss)

        self.chrono.end()

    def show_testset(self, with_embedded_label=True):
        test = load_dataset(args, train=False)

        self.count = 0
        self.acc = 0
        self.cost = 0

        for x, y in test:
            with torch.no_grad():
                if with_embedded_label:
                    x, y = embed_label(x, y)

                x = x.cuda()
                y = y.cuda()

                output = self.model(x)

                loss = self.criterion(output, y.long())
                acc = (output.max(dim=1)[1] == y).sum().item() / len(y)

                self.update_stat(acc, loss)

        print('Test Set [EmbeddedLabel: {}] acc: {acc:.2f}%  loss: {loss:.4f}'.format(
            with_embedded_label,
            acc=self.acc * 100 / self.count,
            loss=self.cost / self.count)
        )


if __name__ == '__main__':
    import sys
    import torch
    import torch.nn as nn
    import torch.nn.parallel

    import torch.optim

    from MixedPrecision.tools.optimizer import OptimizerAdapter
    import MixedPrecision.tools.utils as utils

    from apex.fp16_utils import network_to_half

    sys.stderr = sys.stdout

    parser = get_parser()
    args = parser.parse_args()

    torch.set_num_threads(args.workers)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    utils.set_use_gpu(args.gpu, not args.no_bench_mode)
    utils.set_use_half(args.half)
    utils.show_args(args)

    data_loader = load_dataset(args)

    model = utils.enable_cuda(resnet.resnet18())

    if args.half:
        model = network_to_half(model)

    criterion = utils.enable_cuda(nn.CrossEntropyLoss())

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

    trainer = Experience1a(
        model=model,
        loader=data_loader,
        criterion=criterion,
        optimizer=optimizer
    )

    trainer.epoch_count = args.epochs
    #trainer.train()
    trainer.show_testset(args, True)
    trainer.show_testset(args, False)

    #trainer.report_gpu()
    #trainer.report_train()


from MixedPrecision.tools.args import get_parser
from MixedPrecision.tools.train import Trainer
from MixedPrecision.tools.loaders import load_dataset
from MixedPrecision.tools.utils import throttle

import torch
import torchvision.models.resnet as resnet


class Baseline(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0
        self.acc = 0
        self.cost = 0
        self.model.name = self.name = 'Baseline'

    def after_batch(self, id, batch_context):
        print('[{data[epoch][id]:2d} |{data[id]:4d}] acc: {data[acc]:.2f}%  loss: {data[loss]:.4f}'.format(data=batch_context))

    def after_epoch(self, id, epoch_context):
        print('----')
        print('[{data[id]:2d}] acc: {acc:.2f}%  loss: {loss:.4f}'.format(
            data=epoch_context, acc=self.acc * 100 / self.count, loss=self.cost / self.count))

        self.report_train()
        self.report_gpu()
        print('----')
        self.count = 0
        self.acc = 0
        self.cost = 0

    def update_stat(self, acc, loss, loss2=1):
        self.batch_context['acc'] = acc * 100
        self.batch_context['loss'] = loss
        self.batch_context['loss2'] = loss2

        self.count += 1
        self.cost += loss * loss2
        self.acc += acc

    def train_batch(self):
        self.chrono.start()
        index, (x, y) = self.fetch_input()

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

    def show_testset(self, args, with_embedded_label=True):
        test = load_dataset(args, train=False)

        self.count = 0
        self.acc = 0
        self.cost = 0

        for x, y in test:
            with torch.no_grad():
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

    trainer = Baseline(
        model=model,
        loader=data_loader,
        criterion=criterion,
        optimizer=optimizer
    )

    trainer.train()
    trainer.report_gpu()
    trainer.report_train()


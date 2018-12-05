from MixedPrecision.tools.args import get_parser
from MixedPrecision.tools.train import Trainer
from MixedPrecision.tools.loaders import load_imagenet
from MixedPrecision.tools.utils import throttle

import torchvision.models.resnet as resnet


class Experience1a(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_call = throttle(lambda x: print('{data[acc]:.4f} % {data[loss]:.4f}'.format(data=x)), 10)

    def train_batch(self):
        self.chrono.start()
        index, (x, y) = self.fetch_input()

        # we have 1000 classes the images is 224x224
        x = x.view(-1, 3, 50176)
        x[:, 0, y] = 0
        x = x.view(-1, 3, 224, 224)

        x = x.cuda()
        y = y.cuda()

        self.chrono.start()
        output = self.model(x)
        loss = self.criterion(output, y.long())

        acc = (output.max(dim=1)[1] == y).sum().item() / len(y)
        self.batch_call({'acc': acc, 'loss': loss.item()})

        self.chrono.start()
        self.backward(loss)

        self.chrono.end()


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

    data_loader = load_imagenet(args)

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

    trainer.train()
    trainer.report_gpu()
    trainer.report_train()


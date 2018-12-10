from TruncatedResNet import TruncatedResNet
from baseline import Baseline
import torch
import torch.nn as nn
import torchvision.models.resnet as resnet


class ResnetRegression(nn.Module):

    def __init__(self, output=(224, 224)):
        super(ResnetRegression, self).__init__()
        # self.resizer = nn.Linear(3 * 32 * 32, 3 * 224 * 224)
        # torch.nn.ConvTranspose2d()

        self.rectify_fc2_backward = []
        self.classify_fc_backward = []

        self.resnet = TruncatedResNet(resnet.BasicBlock, [2, 2, 2, 2])

        # torch.Size([64, 512, 7, 7]) => torch.Size([64, 8, 15, 15])
        self.rectify_fc1 = nn.ConvTranspose2d(512, 8, kernel_size=3, stride=2)

        # torch.Size([64, 8, 15, 15]) => 64, 1, 224, 224
        self.rectify_fc2 = nn.Linear(8 * 15 * 15, output[0] * output[1])
        # self.rectify_fc2.register_backward_hook(lambda x: self.rectify_fc2_backward.append(x.mean().item()))

        self.output = output
        self.name = 'baseline_regression'

    def forward(self, x):
        # x = self.resizer(x.view(-1, 3, 32, 32))
        _, last = self.resnet(x)

        x_hat = self.rectify_fc1(last)
        x_hat = self.rectify_fc2(x_hat.view(-1, 8 * 15 * 15))

        return x_hat.view(-1, self.output[0], self.output[1])


class BaselineRegression(Baseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.name = self.name = 'baseline_regression'

    def train_batch(self):
        self.chrono.start()
        index, (x, y) = self.fetch_input()

        x = x.cuda()
        y = y.cuda()

        self.chrono.start()
        output = self.model(x)
        loss = self.criterion(output, x)

        acc = (output.max(dim=1)[1] == y).sum().item() / len(y)
        self.update_stat(acc, loss)

        self.chrono.start()
        self.backward(loss)

        self.chrono.end()


if __name__ == '__main__':

    import sys
    import torch
    import torch.nn as nn
    import torch.nn.parallel

    import torch.optim

    from baseline import Baseline

    from MixedPrecision.tools.args import get_parser
    from MixedPrecision.tools.loaders import load_dataset
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

    model = utils.enable_cuda(ResnetRegression())

    if args.half:
        model = network_to_half(model)

    criterion = utils.enable_cuda(
        nn.L1Loss(reduction='elementwise_mean')
    )

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

    trainer = BaselineRegression(
        model=model,
        loader=data_loader,
        criterion=criterion,
        optimizer=optimizer
    )

    trainer.train()
    trainer.report_gpu()
    trainer.report_train()

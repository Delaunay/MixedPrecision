from MixedPrecision.tools.args import get_parser
from MixedPrecision.tools.loaders import load_dataset
from MixedPrecision.tools.utils import throttle

from TruncatedResNet import TruncatedResNet
from baseline import Baseline

import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import torchvision.transforms as transforms


torch.set_num_threads(12)


class HybridLoss:
    def __init__(self, mode=0.1):
        self.classifier_loss_function = nn.CrossEntropyLoss()
        self.regression_loss_function = nn.L1Loss(reduction='elementwise_mean')
        self.mode = mode
        self.classifier_loss = 0
        self.regression_loss = 0

    def __call__(self, y, y_hat, x, x_hat):
        # x_hat has only one channel

        with torch.no_grad():
            x_gray = x.mean(dim=1)

        # return self.mode * self.classifier_loss(y_hat, y) + (1 - self.mode) * ((x_hat - x_gray) ** 2)
        self.classifier_loss = self.classifier_loss_function(y_hat, y)
        self.regression_loss = self.regression_loss_function(x_hat, x_gray) ** 0.25  #((x_hat - x_gray) ** 2).mean()

        v = self.classifier_loss * self.regression_loss

        self.classifier_loss = self.classifier_loss.item()
        self.regression_loss = self.regression_loss.item()
        return v

    def cuda(self):
        self.classifier_loss_function.cuda()
        return self


class HybridClassifier(nn.Module):

    def __init__(self, classes=1000, output=(224, 224)):
        super(HybridClassifier, self).__init__()
        # self.resizer = nn.Linear(3 * 32 * 32, 3 * 224 * 224)
        # torch.nn.ConvTranspose2d()

        self.rectify_fc2_backward = []
        self.classify_fc_backward = []

        self.resnet = TruncatedResNet(resnet.BasicBlock, [2, 2, 2, 2])
        self.classify_fc = nn.Linear(512 * resnet.BasicBlock.expansion, classes)
        self.classify_fc.register_backward_hook(lambda x: self.classify_fc_backward.append(x.mean().item()))

        # torch.Size([64, 512, 7, 7]) => torch.Size([64, 8, 15, 15])
        self.rectify_fc1 = nn.ConvTranspose2d(512, 8, kernel_size=3, stride=2)
        # torch.Size([64, 8, 15, 15]) => 64, 1, 224, 224
        self.rectify_fc2 = nn.Linear(8 * 15 * 15, output[0] * output[1])
        self.rectify_fc2.register_backward_hook(lambda x: self.rectify_fc2_backward.append(x.mean().item()))

        self.output = output
        self.name = 'Experience1b'

    def forward(self, x):
        # x = self.resizer(x.view(-1, 3, 32, 32))
        out, last = self.resnet(x)
        y_hat = self.classify_fc(out)

        # print(last.shape) = torch.Size([64, 512, 7, 7]) => 25088
        # target => 224x224 => 50176
        x_hat = self.rectify_fc1(last)
        # print(x_hat.shape) => torch.Size([64, 3, 15, 15])

        # x_hat = self.rectify_fc2(x_hat)
        x_hat = self.rectify_fc2(x_hat.view(-1, 8 * 15 * 15))

        return y_hat, x_hat.view(-1, self.output[0], self.output[1])


class Experience1b(Baseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.name = self.name = 'Experience1b'

    def after_batch(self, id, batch_context):
        print('Rectify {} \nClassify {}'.format(self.model.rectify_fc2_backward, self.model.classify_fc_backward))
        self.stats.write('{data[epoch][id]:2d}, {data[id]:4d}, {data[acc]:.2f}, {data[loss]:.4f}, {data[loss2]:.4f}\n'.format(data=batch_context))
        print('[{data[epoch][id]:2d} |{data[id]:4d}] acc: {data[acc]:.2f}%  loss: {data[loss]:.4f} {data[loss2]:.4f}'.format(data=batch_context))

    def train_batch(self):
        self.chrono.start()
        index, (x, y) = self.fetch_input()

        x = x.cuda()
        y = y.cuda()

        self.chrono.start()
        y_hat, x_hat = self.model(x)
        loss = self.criterion(y, y_hat, x, x_hat)

        acc = (y_hat.max(dim=1)[1] == y).sum().item() / len(y)
        self.update_stat(acc, self.criterion.classifier_loss, self.criterion.regression_loss)

        self.chrono.start()
        self.backward(loss)

        self.chrono.end()

    def merge_images(self, a, b):
        import PIL

        images = [a, b]
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = PIL.Image.new('RGB', (total_width, max_height))

        x_offset = 0

        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        return new_im

    def recursive_acc(self, x_hat):
        new_x = torch.tensor((), dtype=torch.float)
        new_x = new_x.new_zeros((64, 3, 224, 224)).cuda()

        for i, img in enumerate(x_hat):
            new_x[i, 0] = img
            new_x[i, 1] = img
            new_x[i, 2] = img

        y_hat2, _ = self.model(new_x)
        return (y_hat2.max(dim=1)[1] == y).sum().item() / len(y)

    def show_testset(self, args, with_embedded_label=True):
        import torchvision.transforms as transforms
        #import matplotlib
        #matplotlib.use('Agg')
        #import matplotlib.pyplot as pl

        test = load_dataset(args, train=True)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.count = 0
        self.acc = 0
        self.cost = 0
        self.acc_recursive = 0
        count = 0

        for index, (x, y) in enumerate(test):
            with torch.no_grad():
                x = x.cuda()
                y = y.cuda()

                y_hat, x_hat = self.model(x)
                loss = self.criterion(y, y_hat, x, x_hat)
                acc = (y_hat.max(dim=1)[1] == y).sum().item() / len(y)
                self.update_stat(acc, loss)

                if index == 0:
                    to_image = transforms.ToPILImage()
                    size = len(x_hat)

                    x_hat = x_hat.cpu().view(size, 1, 224, 224)
                    x = x.cpu().view(size, 3, 224, 224)

                    for i, img in enumerate(x_hat):
                        hat = to_image(img + 0.5)
                        original = to_image(x[i] + 0.5)

                        img = self.merge_images(original, hat)
                        img.save('images/{}_{}.png'.format(i, classes[y[i]]))

        print('Test Set [EmbeddedLabel: {}] acc: {acc:.2f}%  loss: {loss:.4f} acc2: {acc2:.2f}'.format(
            with_embedded_label,
            acc=self.acc * 100 / self.count,
            loss=self.cost / self.count,
            acc2=self.acc_recursive)
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

    data_loader = load_dataset(args, train=True)

    model = utils.enable_cuda(HybridClassifier())

    if args.half:
        model = network_to_half(model)

    criterion = utils.enable_cuda(HybridLoss())

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

    trainer = Experience1b(
        model=model,
        loader=data_loader,
        criterion=criterion,
        optimizer=optimizer
    )

    trainer.epoch_count = args.epochs
    trainer.load_model()
    #trainer.train()
    trainer.show_testset(args, False)




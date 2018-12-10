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



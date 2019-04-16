import time
import torch
import os

from MixedPrecision.tools.chrono import MultiStageChrono
from MixedPrecision.tools.nvidia_smi import make_monitor
from MixedPrecision.tools.utils import throttle


class Trainer:
    def __init__(self, model, criterion, loader, optimizer):
        self.model = model
        self.criterion = criterion
        self.loader = loader
        self.optimizer = optimizer
        self.chrono = MultiStageChrono(['fetch_input', 'forward', 'backward'])
        self.epoch_count = 10
        self.gpu_monitor = None
        self.next_inputs = None
        self.current_iterator = None
        self.batch_context = {}
        self.batch_call = throttle(lambda id, ctx: self.after_batch(id, ctx), 10)
        self.epoch_context = {}
        self.epoch_call = lambda id, ctx: self.after_epoch(id, ctx)
        self.save_model_call = throttle(lambda id, ctx: self.save_model(id, ctx), 20)
        self.batch_id = -1
        self.previous_checkpoint = None
        self.stats = open('training.dat', 'w+')
        self.stats.write('epoch, batch, acc, loss\n')

    def save_model(self, id, ctx):
        name = '{}_{}'.format(self.model.name, id)
        torch.save(self.model.state_dict(), name)

        if self.previous_checkpoint is not None:
            os.remove(self.previous_checkpoint)

        self.previous_checkpoint = name
        file = open('checkpoint_{}'.format(self.model.name), 'w')
        file.write(name)
        file.close()

    def load_model(self):
        try:
            file = open('checkpoint_{}'.format(self.model.name), 'r')
            name = file.read()
            file.close()
            self.model.load_state_dict(torch.load(name))
            print('Model `{}` was loaded from memory'.format(self.model.name))
        except FileNotFoundError:
            print('No Checkpoint found for {}'.format(self.model.name))

    def has_next(self):
        return self.next_inputs is not None

    def forward(self, x, y):
        output = self.model(x)
        return self.criterion(output, y.long())

    def backward(self, loss):
        self.optimizer.zero_grad()
        self.optimizer.backward(loss)
        self.optimizer.step()

    def _next(self):
        try:
            return next(self.current_iterator)
        except StopIteration:
            return None
        except Exception as e:
            print(e)
            return None

    def fetch_input(self):
        self.batch_id, (x, y) = self.next_inputs
        self.next_inputs = self._next()
        return self.batch_id, (x, y)

    def train_batch(self):
        self.chrono.start()
        index, (x, y) = self.fetch_input()

        # Stage 0
        self.chrono.start()
        loss = self.forward(x, y)

        # Stage 1
        self.chrono.start()
        self.backward(loss)

        # Stage 3
        self.chrono.end()

    def after_batch(self, id, batch_context):
        pass

    def after_epoch(self, id, epoch_context):
        pass

    def train_epoch(self):
        self.current_iterator = enumerate(self.loader)
        self.next_inputs = self._next()
        id = 0

        while self.has_next():
            try:
                self.batch_context = {
                    'epoch': self.epoch_context,
                    'id': id
                }
                self.train_batch()
                self.batch_call(id, self.batch_context)
                self.save_model_call(id, self.batch_context)
                id += 1
            except Exception as e:
                print(e)

    def accuracy(self, loader):
        acc = 0
        n = len(loader)
        for x, y in loader:
            t = self.model(x)
            acc += (t.max(dim=1)[1] == y).sum()

        return acc / n

    def train(self):
        monitor_proc, self.gpu_monitor = make_monitor(loop_interval=250)

        try:
            for epoch in range(0, self.epoch_count):
                self.epoch_context = {
                    'id': epoch
                }
                self.train_epoch()
                self.epoch_call(epoch, self.epoch_context)

        finally:
            monitor_proc.terminate()
        self.stats.close()

    def report_train(self):
        self.chrono.report()

    def report_gpu(self):
        if self.gpu_monitor is not None:
            self.gpu_monitor.report()


if __name__ == '__main__':
    t = Trainer(None, None, [], None)

    t.report_gpu()
    t.report_train()






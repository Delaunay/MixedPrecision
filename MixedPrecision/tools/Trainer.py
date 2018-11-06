import time

from MixedPrecision.tools.chrono import MultiStageChrono
from MixedPrecision.tools.nvidia_smi import make_monitor


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
        self._next()

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
            self.next_inputs = next(self.loader)
        except StopIteration:
            self.next_inputs = None
        except Exception as e:
            print(e)
            self.next_inputs = None

    def fetch_input(self):
        inputs = self.next_inputs
        self._next()
        return inputs

    def train_batch(self):
        self.chrono.start()
        x, y = self.fetch_input()

        self.chrono.start()
        loss = self.forward(x, y)

        self.chrono.start()
        self.backward(loss)

        self.chrono.end()

    def train_epoch(self):
        while self.has_next():
            self.train_batch()

    def train(self):
        monitor_proc, self.gpu_monitor = make_monitor(loop_interval=250)

        try:
            for epoch in range(0, self.epoch_count):
                self.train_epoch()

        finally:
            monitor_proc.terminate()

    def report_train(self):
        self.chrono.report()

    def report_gpu(self):
        if self.gpu_monitor is not None:
            self.gpu_monitor.report()


if __name__ == '__main__':
    t = Trainer(None, None, [], None)

    t.report_gpu()
    t.report_train()






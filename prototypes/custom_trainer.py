import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import array
import time
import math

from MixedPrecision.pytorch.dtypes import *
from benchutils.statstream import StatStream

from typing import Tuple


class OptimizerAdapter:
    def __init__(self, optimizer: optim.Optimizer, **kwargs):
        self.optimizer = optimizer

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def step(self, cost=None, closure=None, **kwargs):
        return self.optimizer.step(closure)

    def backward(self, cost: nn.Module):
        return cost.backward()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, img):
        return self.optimizer.load_state_dict(img)


class Fp16OptimizerAdapter:
    def __init__(self, optimizer: optim.Optimizer, **kwargs):
        from apex import amp

        self.optimizer = optimizer
        self.amp = amp.init()

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def step(self, cost=None, closure=None, **kwargs):
        return self.optimizer.step(closure)

    def backward(self, cost: nn.Module):
        with self.amp.scale_loss(cost, self.optimizer) as loss:
            return loss.backward()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, img):
        return self.optimizer.load_state_dict(img)


class Trainer:
    def __init__(self):
        pass

    def fit(self, epoch, dataloader, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass

"""
b 	boolean
i 	signed integer
u 	unsigned integer
f 	floating-point
c 	complex floating-point
m 	timedelta
M 	datetime
O 	object
S 	(byte-)string
U 	Unicode
V 	void
"""


class RingBuffer:
    types = {
        torch.float16: 'f',  # 4
        torch.float32: 'f',  # 4
        torch.float64: 'd',  # 8

        torch.int8:  'b',    # 1
        torch.int16: 'h',    # 2
        torch.int32: 'l',    # 4
        torch.int64: 'q',    # 8

        torch.uint8:  'B',   # 1
        # torch.uint16: 'H',   # 2
        # torch.uint32: 'L',   # 4
        # torch.uint64: 'Q',   # 8
    }

    def __init__(self, size, dtype, default_val=0):
        self.array = array.array(self.types[dtype], [default_val] * size)
        self.capacity = size
        self.offset = 0

    def __getitem__(self, item):
        return self.array[item % self.capacity]

    def __setitem__(self, item, value):
        self.array[item % self.capacity] = value

    def append(self, item):
        self.array[self.offset % self.capacity] = item
        self.offset += 1

    def to_list(self):
        if self.offset < self.capacity:
            return list(self.array[:self.offset])
        else:
            end_idx = self.offset % self.capacity
            return list(self.array[end_idx: self.capacity]) + list(self.array[0:end_idx])

    def __len__(self):
        return min(self.capacity, self.offset)


class Logger:
    def __init__(self, epoch_count=100, batch_count=6):
        self.batch_fmt_size = int(math.log10(batch_count)) + 1

        self.epoch_fmt = f'{int(math.log10(epoch_count)) + 1}d'
        self.batch_fmt = f'{self.batch_fmt_size}d'

        self.epoch_count = epoch_count
        self.batch_count = batch_count

        self.total_batch_count = epoch_count * batch_count

    @staticmethod
    def get_avg(stream: StatStream):
        avg = stream.avg
        if avg == 0:
            avg = stream.val
        return avg

    def compute_eta(self, epoch, batch_id, batch_time):
        return (self.total_batch_count - (epoch * self.batch_count + batch_id)) * self.get_avg(batch_time) / 60

    def _epoch_progress(self, epoch):
        # returns [   0/100]
        return f'[{epoch:{self.epoch_fmt}}/{self.epoch_count:{self.epoch_fmt}}]'

    def _batch_progress(self, batch_id=None):
        if batch_id is None:
            return f"[{' ' * (self.batch_fmt_size * 2 + 1)}]"

        # return  [    0/1234]
        return f'[{batch_id:{self.batch_fmt}}/{self.batch_count:{self.batch_fmt}}]'

    def log_step(self, epoch, batch_id, batch_loss, batch_time, **kwargs):
        ETA = self.compute_eta(epoch, batch_id, batch_time)

        epoch_p = self._epoch_progress(epoch)
        batch_p = self._batch_progress(batch_id)

        print(f'{epoch_p}{batch_p} loss={batch_loss:.6f} ETA={ETA:6.4f} min')

    def log_epoch(self, epoch, loss, **kwargs):
        epoch_p = self._epoch_progress(epoch)
        batch_p = self._batch_progress()

        print(f'{epoch_p}{batch_p} loss={loss:.6f}')

    def log_train(self, train_time, **kwargs):
        print(f'Trained in {train_time / 60:.2f} min')

    def log_metric(self, metric, value):
        pass


class TrainClassifier(Trainer):

    def __init__(self, optimizer: optim.Optimizer, criterion: nn.Module, model: nn.Module, logger: Logger = Logger()):
        super(TrainClassifier, self).__init__()

        if not issubclass(type(optimizer), OptimizerAdapter):
            optimizer = OptimizerAdapter(optimizer)

        self.optimizer = optimizer
        self.criterion = criterion
        self.model = model
        self.logger = logger

        self.epoch_time = StatStream(drop_first_obs=10)
        self.batch_time = StatStream(drop_first_obs=10)
        self.batch_count = 0

    def fit(self, epochs: int, dataloader: data.dataloader, *args, **kwargs):
        train_start = time.time()
        data_set_size = len(dataloader)
        epoch_time = self.epoch_time
        batch_time = self.batch_time
        self.model.train()

        for epoch in range(epochs):
            epoch_start = time.time()

            loss = 0
            loading_start = time.time()

            for batch_id, batch in enumerate(dataloader):
                self.batch_count += 1

                loading_end = time.time()
                loading_time = loading_end - loading_start

                loss += TrainClassifier._step(**locals())
                loading_start = time.time()

            epoch_end = time.time()
            epoch_time.update(epoch_end - epoch_start)

            self.logger.log_metric('loss', loss)

            # ---
            v = locals()
            v.pop('self')
            self.logger.log_epoch(**v)
            # ---

        train_end = time.time()
        train_time = train_end - train_start

        # ---
        v = locals()
        v.pop('self')
        self.logger.log_train(**v)
        # ---

    def _step(self, batch: (Tensor[NCHW], Tensor[N]), loading_time, batch_time, **kwargs):
        compute_start = time.time()
        self.model.train()
        batch_loss = self.step(batch)

        compute_end = time.time()
        compute_time = compute_end - compute_start

        batch_time.update(loading_time + compute_time)

        # ---
        scope = self._merge_env(locals(), kwargs)
        scope.pop('self')
        self.logger.log_step(**scope)
        # ---

        return batch_loss

    def step(self, batch: (Tensor[NCHW], Tensor[N])):
        data, label = batch

        self.optimizer.zero_grad()

        p_label = self.model(data)
        oloss = self.criterion(p_label, label)
        batch_loss = oloss.item()

        # compute gradient and do SGD step
        self.optimizer.backward(oloss)
        self.optimizer.step()
        return batch_loss

    @staticmethod
    def _merge_env(scope, env):
        for k, v in env.items():
            scope[k] = v
        return scope

    def resume(self, path, find_latest=False):
        import glob

        if find_latest:
            stuff = path.split('.')
            path = '.'.join(stuff[:-1])

            files = list(glob.glob(f'{path}*'))
            files.sort()

            path = files[-1]

        state = torch.load(path)
        self.optimizer.load_state_dict(state['optimizer'])
        self.model.load_state_dict(state['model'])
        self.batch_count = state['batch_count']
        self.epoch_time = StatStream.from_dict(state['epoch_time'])
        self.batch_time = StatStream.from_dict(state['batch_time'])

    def save(self, path, override=True):
        import glob

        stuff = path.split('.')
        ext = stuff[-1]
        path = '.'.join(stuff[:-1])

        if not override:
            n = len(glob.glob(f'{path}*'))
            if n > 0:
                path = f'{path}_{self.batch_count}_{n}'

        path = f'{path}.{ext}'

        torch.save({
            'batch_count': self.batch_count,
            'optimizer': self.optimizer.state_dict(),
            'model': self.model.state_dict(),
            'epoch_time': self.epoch_time.state_dict(),
            'batch_time': self.batch_time.state_dict()
        }, path)


def test():
    import os
    from torchvision.models.resnet import resnet18

    model = resnet18().cuda()
    cost = nn.CrossEntropyLoss().cuda()

    trainer = TrainClassifier(
        optimizer=optim.SGD(params=model.parameters(), lr=0.01),
        criterion=cost,
        model=model
    )

    save_path = 'data.ts'

    if os.path.exists(save_path):
        trainer.resume(save_path, find_latest=True)

    trainer.fit(
        100,
        [(torch.rand(1, 3, 224, 224).cuda(), torch.rand(1).long().cuda()),
         (torch.rand(1, 3, 224, 224).cuda(), torch.rand(1).long().cuda()),
         (torch.rand(1, 3, 224, 224).cuda(), torch.rand(1).long().cuda()),
         (torch.rand(1, 3, 224, 224).cuda(), torch.rand(1).long().cuda()),
         (torch.rand(1, 3, 224, 224).cuda(), torch.rand(1).long().cuda()),
         (torch.rand(1, 3, 224, 224).cuda(), torch.rand(1).long().cuda())
         ]
    )

    trainer.save(save_path, override=False)


if __name__ == '__main__':
    import sys
    sys.stderr = sys.stdout

    test()

    #db = RoundRobinDatabase(10, torch.int32)

    # for i in range(5):
    #     db.append(i)
    #
    # print(db.to_list())
    #
    # for i in range(20):
    #     db.append(i)
    #
    # print(db.to_list())

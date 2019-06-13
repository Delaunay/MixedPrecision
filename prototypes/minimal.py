import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from benchutils.chrono import show_eta, MultiStageChrono, time_this

from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


chrono = MultiStageChrono()


class CNNBase(nn.Module):
    def __init__(self, num_inputs, hidden_size=512):
        super(CNNBase, self).__init__()
        self._hidden_size = hidden_size

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU()
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)
        return self.critic_linear(x), x, rnn_hxs

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def recurrent_hidden_state_size(self):
        return 1

    @property
    def is_recurrent(self):
        return False

def test_cnn_base():
    # torch.Size([2, 4, 84, 84])

    device = torch.device('cuda')
    print(torch.cuda.get_device_name(device))

    print(CNNBase((4, 84, 84)[0], hidden_size=512))

    prev = None
    for batch_size in [2, 4, 8, 16, 32, 64, 128]:
        chrono = MultiStageChrono()
        repeat = 30
        exp = f'forward_{batch_size}'

        input = torch.rand(batch_size, 4, 84, 84).cuda()

        net = CNNBase((4, 84, 84)[0], hidden_size=512)
        net.cuda()

        for _ in range(0, 10):
            with chrono.time(exp):
                for _ in range(0, repeat):
                    net(input, None, None)
                torch.cuda.synchronize()

        for _ in range(0, 30):
            with chrono.time(exp):
                for _ in range(0, repeat):
                    net(input, None, None)
                torch.cuda.synchronize()

        speed = batch_size * repeat / chrono.chronos[exp].avg
        speed_up = ''
        if prev:
            speed_up = f'Speed up x{speed / prev:7.4f}'

        print(f'{exp:>30} {speed:12,.4f} item/sec {speed_up}')
        prev = speed


if __name__ == '__main__':
    # torch.Size([2, 4, 84, 84])

    device = torch.device('cuda')
    print(torch.cuda.get_device_name(device))

    input_size = 4 * 84 * 84
    hidden_size = 512
    output_size = 1000


    def make_linear_model(batch):
        return nn.Sequential(
            Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )

    print(make_linear_model(4))

    prev = None
    for batch_size in [2, 4, 8, 16, 32, 64, 128]:
        chrono = MultiStageChrono()
        repeat = 30
        exp = f'forward_{batch_size}'

        input = torch.rand(batch_size, 4, 84, 84).cuda()
        net = make_linear_model(batch_size).cuda()

        for _ in range(0, 10):
            with chrono.time(exp):
                for _ in range(0, repeat):
                    net(input)
                torch.cuda.synchronize()

        for _ in range(0, 30):
            with chrono.time(exp):
                for _ in range(0, repeat):
                    net(input)
                torch.cuda.synchronize()

        speed = batch_size * repeat / chrono.chronos[exp].avg
        speed_up = ''
        if prev:
            speed_up = f'Speed up x{speed / prev:7.4f}'

        print(f'{exp:>30} {speed:12,.4f} item/sec {speed_up}')
        prev = speed


import torch
import torch.nn as nn
import time

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


if __name__ == '__main__':
    # torch.Size([2, 4, 84, 84])

    device = torch.device('cuda')
    print(torch.cuda.get_device_name(device))

    input_size = 4 * 84 * 84
    hidden_size = 512
    output_size = 1000

    def make_linear_model():
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

    print(make_linear_model())

    prev = None
    for batch_size in [2, 4, 8, 16, 32, 64, 128]:

        repeat = 30
        count = 30
        exp = f'forward_{batch_size}'

        input = torch.rand(batch_size, 4, 84, 84).cuda()
        net = make_linear_model().cuda()

        for _ in range(0, 10):
            for _ in range(0, repeat):
                net(input)

        a = 0
        for _ in range(0, count):
            s = time.time()
            for _ in range(0, repeat):
                net(input)
            torch.cuda.synchronize()
            e = time.time()
            a += e - s

        speed = batch_size * repeat * count / a
        speed_up = ''
        if prev:
            speed_up = f'Speed up x{speed / prev:7.4f}'

        print(f'{exp:>30} {speed:12,.4f} item/sec {speed_up}')
        prev = speed

"""
GeForce GTX 1060 6GB

                     forward_2   2,083.1906 item/sec 
                     forward_4   4,247.2901 item/sec Speed up x 2.0388
                     forward_8  12,406.9708 item/sec Speed up x 2.9211
                    forward_16  24,555.4173 item/sec Speed up x 1.9792
                    forward_32  47,116.5335 item/sec Speed up x 1.9188
                    forward_64  60,734.9503 item/sec Speed up x 1.2890
                   forward_128  80,558.9667 item/sec Speed up x 1.326
                   
GeForce RTX 2080 Ti
                     forward_2   3,746.0180 item/sec 
                     forward_4   8,115.0933 item/sec Speed up x 2.1663
                     forward_8  16,550.8645 item/sec Speed up x 2.0395
                    forward_16  30,116.2594 item/sec Speed up x 1.8196
                    forward_32  59,877.0770 item/sec Speed up x 1.9882
                    forward_64 121,918.6430 item/sec Speed up x 2.0361
                   forward_128 164,930.3190 item/sec Speed up x 1.3528

Ellesmere [Radeon RX 470/480/570/570X/580/580X]

                     forward_2     197.6428 item/sec 
                     forward_4     394.5175 item/sec Speed up x 1.9961
                     forward_8     779.3699 item/sec Speed up x 1.9755
                    forward_16   1,524.8228 item/sec Speed up x 1.9565
                    forward_32   2,981.2282 item/sec Speed up x 1.9551
                    forward_64   5,816.8904 item/sec Speed up x 1.9512
                   forward_128  11,619.5541 item/sec Speed up x 1.9976
            
Vega 20

                     forward_2     298.6409 item/sec 
                     forward_4     600.7616 item/sec Speed up x 2.0117
                     forward_8   1,190.5813 item/sec Speed up x 1.9818
                    forward_16   2,326.0575 item/sec Speed up x 1.9537
                    forward_32   4,424.8875 item/sec Speed up x 1.9023
                    forward_64   8,642.0447 item/sec Speed up x 1.9531
                   forward_128  17,367.4644 item/sec Speed up x 2.0096

"""
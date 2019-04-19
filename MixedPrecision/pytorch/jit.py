import torch
import torch.nn as nn
import torch.nn.parallel

import torch.optim

import torch.utils.data
import torch.utils.data.distributed

import torchvision.models.resnet as resnet
import time

model = resnet.resnet18().cuda()

print('Tracing')
trace = torch.jit.trace(model, (torch.rand(64, 3, 224, 224),), optimize=True)

trace.save('resnet18.pt')
print('---')

print(trace.graph)

data = torch.jit.load('resnet18.pt')

input = torch.rand(64, 3, 224, 224)

s = 0
s2 = 0

for i in range(0, 30):
    t = time.time()
    data(input)
    s = time.time() - t

    t = time.time()
    model(input)
    s2 = time.time() - t

print('JIT {}'.format(s))
print('Python: {}'.format(s2))

s = 0
s2 = 0

for i in range(0, 30):
    t = time.time()
    data(input)
    s = time.time() - t

    t = time.time()
    model(input)
    s2 = time.time() - t

print('JIT {}'.format(s))
print('Python: {}'.format(s2))

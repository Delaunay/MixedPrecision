import torch
import time
import pandas as pd
import argparse
import itertools

input_size = [2, 4, 8, 10, 16, 32, 64, 100, 128, 200, 256, 500, 512]
hidden_size = [500, 512, 100, 1024, 2000, 2048]
output_size = [10, 500, 512, 100, 1024, 2000, 2048]

parser = argparse.ArgumentParser()
parser.add_argument('--input-size', type=int, nargs='*', default=input_size)
parser.add_argument('--hidden-size', type=int, nargs='*', default=hidden_size)
parser.add_argument('--output-size', type=int, nargs='*', default=output_size)
parser.add_argument('--repeat', type=int, default=1000)
parser.add_argument('--csv', type=str, default=None, help='tensile csv file'
    'https://github.com/ROCmSoftwarePlatform/Tensile/blob/develop/Tensile/Configs/deep_bench_nn.csv')

args = parser.parse_args()

input_size = args.input_size
hidden_size = args.hidden_size
output_size = args.output_size

configs = itertools.product(input_size, hidden_size, output_size)

if args.csv:
    csv = open(args.csv, 'r').read().split('\n')
    csv = map(lambda row: row.strip(), csv)
    csv = filter(lambda row: len(row) > 0, csv)
    configs = map(lambda row: map(lambda val: int(val), row.split(',')), csv)

device = torch.device('cuda')
print(torch.cuda.get_device_name(device))

prev = None

timings = []

all = time.time()
# batch size
for idx, (b, h, o) in enumerate(configs):
    input = torch.rand(b, h).cuda()
    weight = torch.rand(h, o).cuda()

    for _ in range(0, 5):
        out = input.mm(weight)
    torch.cuda.synchronize()

    s = time.time()
    for _ in range(0, args.repeat):
        out = input.mm(weight)

    torch.cuda.synchronize()
    e = time.time()

    t = e - s
    timings.append([b, h, o, t])
    print(f'{idx:6d}) b-{b:6d} h-{h:6d} o-{o:6d} {t:8.4f} s')

end = time.time()

print(f'Ran everything in {end - all} s')

df = pd.DataFrame(timings)
gpu = (torch.cuda.get_device_name(device)
       .replace(" ", "_")
       .replace("/", "")
       .replace("[", "")
       .replace(']', ""))
df.columns = ['b', 'h', 'o', 't']
df.to_csv(f'{gpu}.csv')

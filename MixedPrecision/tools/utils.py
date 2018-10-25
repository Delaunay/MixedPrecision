import torch
import torch.utils.data.dataloader

import time
from MixedPrecision.tools.stats import StatStream

global_use_gpu = False
global_use_half = False


def set_use_gpu(val, benchmode=True):
    global global_use_gpu

    # enable the inbuilt cudnn auto-tuner
    # Faster for fixed size inputs
    if benchmode:
        torch.backends.cudnn.benchmark = True

    global_use_gpu = torch.cuda.is_available() and val


def set_use_half(val):
    global global_use_half
    global_use_half = val

    if val:
        import torch.backends.cudnn as cudnn
        assert cudnn.CUDNN_TENSOR_OP_MATH == 1, 'fp16 mode requires to be compiled with TC enabled'
        assert cudnn.enabled, 'fp16 mode requires cudnn backend to be enabled.'

        #  CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION == 1?
        # CUDNN_TENSOR_NCHW_VECT_C


def use_gpu() -> bool:
    global global_use_gpu
    return global_use_gpu


def use_half() -> bool:
    global global_use_gpu
    global global_use_half

    return global_use_gpu and global_use_half


def enable_cuda(object):
    if use_gpu():
        return object.cuda()

    return object


def enable_half(object):
    if not use_gpu():
        return object

    if not use_half():
        return object.cuda()

    # F32 Tensor
    try:
        if object.dtype == torch.float32:
            return object.cuda(non_blocking=True).half(non_blocking=True)
    except:
        # Not a tensor
        return object.cuda().half()

    # different type
    return object.cuda(non_blocking=True)


def summary(model, input_size, batch_size):
    try:
        import torchsummary
        torchsummary.summary(model, input_size, batch_size=batch_size)
    except Exception as e:
        print(e)
        pass


def fast_collate(batch):
    """
        from Apex by NVIDIA
    """
    import numpy as np

    imgs = [img[0] for img in batch]

    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)

    w = imgs[0].size[0]
    h = imgs[0].size[1]

    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)

    for i, img in enumerate(imgs):

        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)

        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)

        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets


class TimedCollate:
    def __init__(self, collate, time_stream=StatStream(10)):
        self.collate = collate
        self.time_stream = time_stream

    def __call__(self, batch):
        start = time.time()
        v = self.collate(batch)
        end = time.time()
        self.time_stream += end - start
        return v


timed_fast_collate = TimedCollate(collate=fast_collate)
timed_default_collate = TimedCollate(collate=torch.utils.data.dataloader.default_collate)


def bench_collate(collate):
    import MixedPrecision.tools.report as report
    from PIL import Image

    x = Image.new('RGB', (224, 224), color = 'red')
    y = 1
    bs = 256
    data = [[x, y] for _ in range(0, bs)]
    batch = None

    for _ in range(0, 100):
        batch = collate(data)

    timed = collate.time_stream
    print(collate.time_stream.to_array())
    report.print_table(
        ['Metric', 'Average', 'Deviation', 'Min', 'Max', 'count'],
        [
            ['collate_time'] + timed.to_array(),
            ['collate_speed', bs / timed.avg, 'NA', bs / timed.max, bs / timed.min, timed.count]
        ]
    )


if __name__ == '__main__':
    bench_collate(timed_fast_collate)
    bench_collate(timed_default_collate)



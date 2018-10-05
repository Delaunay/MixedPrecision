import torch


global_use_gpu = False
global_use_half = False


def set_use_gpu(val):
    global global_use_gpu
    global_use_gpu = torch.cuda.is_available() and val


def set_use_half(val):
    global global_use_half
    global_use_half = val

    if val:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."


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


def summary(model, input_size):
    try:
        import torchsummary
        torchsummary.summary(model, input_size, device='cuda' if use_gpu() else 'cpu')
    except:
        pass
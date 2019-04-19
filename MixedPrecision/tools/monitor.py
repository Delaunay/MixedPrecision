from MixedPrecision.tools.nvidia_smi import NvGpuMonitor
from MixedPrecision.tools.rocm_smi import AmdGpuMonitor
from MixedPrecision.tools.utils import get_device_vendor, AMD, NVIDIA

from typing import *
from multiprocessing import Process


class GpuMonitor:
    def run(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def report(self):
        raise NotImplementedError

    def arrays(self, common):
        raise NotImplementedError


def start_monitor(monitor: GpuMonitor) -> GpuMonitor:
    monitor.run()
    return monitor


def make_monitor(loop_interval=1000, device_id=0) -> Tuple[Process, GpuMonitor]:
    vendor = get_device_vendor()

    if vendor is AMD:
        monitor = AmdGpuMonitor(loop_interval, device_id)
    else:
        monitor = NvGpuMonitor(loop_interval, device_id)

    proc = Process(target=start_monitor, args=(monitor,))
    proc.start()
    return proc, monitor


class GpuMonitorCtx:
    def __init__(self, loop_interval=1000, device_id=0):
        self.loop = loop_interval
        self.device = device_id
        self.proc = None
        self.monitor = None

    def __enter__(self):
        self.proc, self.monitor = make_monitor(self.loop, self.device)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.stop()
        self.proc.terminate()

import torch
import time

import MixedPrecision.tools.utils as utils
from MixedPrecision.tools.stats import StatStream


class DataPreFetcher:
    """
        Adapted From Apex from NVIDIA

        Prefetch data async, normalize it, and send it to cuda in the correct format (f16 or f32)
    """
    def __init__(self, loader, mean , std, cpu_stats=StatStream(0), gpu_stats=StatStream(0)):
        self.loader = iter(loader)
        self.mean = mean
        self.std = std
        self.next_target = None
        self.next_input = None
        self.stream = self._make_stream()

        self.start_event = torch.cuda.Event(enable_timing=True, blocking=False, interprocess=False)
        self.end_event = torch.cuda.Event(enable_timing=True, blocking=False, interprocess=False)

        self.gpu_time = gpu_stats
        self.cpu_time = cpu_stats
        self.preload()

    @staticmethod
    def _make_stream():
        if utils.use_gpu():
            return torch.cuda.Stream()
        return None

    def preload(self):
        """
            Load data from Pytorch Dataloader and asynchronously send it to the device
        """
        try:
            start = time.time()
            self.next_input, self.next_target = next(self.loader)
            end = time.time()
            self.cpu_time.update(end - start)

        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.stream.record_event(self.start_event)

            self.next_input = self.next_input.cuda(async=True)
            self.next_target = self.next_target.cuda(async=True)

            if utils.use_half():
                self.next_input = self.next_input.half()
            else:
                self.next_input = self.next_input.float()

            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            self.stream.record_event(self.end_event)

    def next(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)

            # Compute the time it took to load the data
            self.end_event.synchronize()
            self.gpu_time += self.start_event.elapsed_time(self.end_event)

        input = self.next_input
        target = self.next_target

        # Start fetching next
        self.preload()
        return input, target

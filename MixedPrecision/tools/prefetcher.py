import torch
import MixedPrecision.tools.utils as utils


class DataPreFetcher:
    """
        Adapted From Apex from NVIDIA

        Prefetch data async, normalize it, and send it to cuda in the correct format (f16 or f32)
    """
    def __init__(self, loader, mean , std):
        self.loader = iter(loader)
        self.mean = mean
        self.std = std
        self.next_target = None
        self.next_input = None
        self.stream = self._make_stream()
        self.preload()

    def _make_stream(self):
        if utils.use_gpu():
            return torch.cuda.Stream()
        return None

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(async=True)
            self.next_target = self.next_target.cuda(async=True)

            if utils.use_half():
                self.next_input = utils.enable_half(self.next_input)
            else:
                self.next_input = self.next_input.float()

            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)

        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

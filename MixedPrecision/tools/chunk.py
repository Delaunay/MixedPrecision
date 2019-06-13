
# Not really improving loading speed
class BatchChunksIterator:
    """ Pytorch dataloader works better with big batch sizes.
        so we allow the user to specify a bigger batch size for the dataloader only.
        but the input size will be smaller.

        [In] < ./image_classification/convnets/pytorch/run.sh /Tmp/mlperf/data//data/ImageNet/train/ --repeat 20\
            --number 5 --workers 8 --arch resnet18 --cuda --opt-level O0 --batch-size 64 --loader-batch-size 64

        [Out]>   "avg": 384.77666779370503 img/sec :(

    """

    def __init__(self, dataloader: DataLoader, batch_size_input,):
        self.loader = iter(dataloader)
        self.bs_loader = dataloader.batch_size
        self.bs_input = batch_size_input
        self.disabled = False

        # The loader should have a bigger batch size for this to matter
        if self.bs_loader <= self.bs_input:
            self.disabled = True
            print('/!\\ BatchChunkIterator is disabled')

        self.bs_count = self.bs_loader // self.bs_input
        assert self.bs_loader % self.bs_input == 0

        self.current_input = None
        self.current_target = None
        self.i = self.bs_count

    def __iter__(self):
        return self

    def __next__(self):
        if self.disabled:
            return next(self.loader)

        if self.i == self.bs_count:
            self.i = 0
            self.current_input, self.current_target = next(self.loader)

        start = self.i * self.bs_input
        end = start + self.bs_input
        self.i += 1

        return self.current_input[start:end, :], self.current_target[start:end]

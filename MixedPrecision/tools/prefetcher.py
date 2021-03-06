import torch
import torch.multiprocessing as multiprocessing
import threading
import time

import MixedPrecision.tools.utils as utils
from MixedPrecision.tools.stats import StatStream


class DataPreFetcher:
    """
        Adapted From Apex from NVIDIA

        Prefetch data 'async', normalize it, and send it to cuda in the correct format (f16 or f32)
        the most time is passed on the CPU so this prefetcher is not great since CPU = sync code
        the only async part is the GPU part and it is negligible
        I think the Async one is going to be better
    """
    def __init__(self, loader, mean , std, cpu_stats=StatStream(0), gpu_stats=StatStream(0)):
        print('Using Prefetcher')
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

    @property
    def read_timer(self):
        if hasattr(self.loader, 'read_timer'):
            return self.loader.read_timer
        return None

    @property
    def transform_timer(self):
        if hasattr(self.loader, 'read_timer'):
            return self.loader.transform_timer
        return None

    @property
    def load_timer(self):
        return self.cpu_time

    @property
    def gpu_timer(self):
        return self.gpu_time

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

            self.next_input = self.next_input.cuda()
            self.next_target = self.next_target.cuda()

            if utils.use_half():
                self.next_input = self.next_input.half()
            else:
                self.next_input = self.next_input.float()

            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            self.stream.record_event(self.end_event)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)

            # Compute the time it took to load the data
            self.end_event.synchronize()
            self.gpu_time += self.start_event.elapsed_time(self.end_event) / 1000.0

        input = self.next_input
        target = self.next_target

        # Start fetching next
        self.preload()
        return input, target


def prefetch(work, results, loader, stats):
    import torch
    import random

    torch.set_num_threads(1)
    random.seed(0)
    torch.manual_seed(0)

    while True:
        message = work.get()

        if message == 'next':
            try:
                s = time.time()
                results.put(next(loader))
                stats += time.time() - s
            except StopIteration:
                results.put(None)
                break
            except Exception as e:
                print(e)
                raise e

        if message == 'stop':
            break


class AsyncPrefetcherWorker(threading.Thread):
    def __init__(self, work, results, loader):
        super().__init__(name='AsyncPrefetcherWorker')

        self.work = work
        self.results = results
        self.loader = loader
        self.stat = StatStream(10)

    def run(self):
        prefetch(self.work, self.results, self.loader, self.stat)


class AsyncPrefetcher:
    """
        We cannot use multiprocessing here because the DataLoader class is not sharable across processes.
        So we implement the async behaviour using a Thread.
        Threads in python are concurrent not parallel. So we hope this will still enable python
        to prefetch the data while the model is computed on the GPU

    """
    def __init__(self, loader, buffering=2):
        self.loader = iter(loader)
        self.data = None
        self.loading_stat = StatStream(1)
        self.wait_time = StatStream(1)
        #self.manager = Manager()
        #self.work_queue = self.manager.Queue()
        #self.result_queue = self.manager.Queue()

        self.work_queue = multiprocessing.SimpleQueue()
        self.result_queue = multiprocessing.SimpleQueue()

        # MP implementation
        #self.worker = multiprocessing.Process(target=prefetch, args=(self.work_queue, self.result_queue, self.loader, self.loading_stat))
        #self.worker.start()

        self.worker = AsyncPrefetcherWorker(self.work_queue, self.result_queue, self.loader)
        self.worker.start()

        # put n batch in advance
        for i in range(buffering):
            self.work_queue.put('next')

    @property
    def read_timer(self):
        if hasattr(self.loader, 'read_timer'):
            return self.loader.read_timer
        return None

    @property
    def transform_timer(self):
        if hasattr(self.loader, 'read_timer'):
            return self.loader.transform_timer
        return None

    @property
    def wait_timer(self):
        return self.wait_time

    def preload(self):
        if self.worker.is_alive():
            self.work_queue.put('next')

    def __iter__(self):
        return self

    def __next__(self):
        s = time.time()
        data = self.result_queue.get()
        self.preload()

        if data is None:
            raise StopIteration
        self.wait_time += time.time() - s
        return data[0].cuda(), data[1].cuda()

    next = __next__

    def __del__(self):
        self.stop()
        #self.work_queue.close()
        #self.result_queue.close()
        #self.work_queue.join_thread()

    def stop(self):
        self.work_queue.put('stop')
        self.worker.join(10)

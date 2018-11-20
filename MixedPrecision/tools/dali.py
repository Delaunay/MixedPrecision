try:
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

from MixedPrecision.tools.folder import make_dali_cached_file_list_which_is_also_a_file


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, half=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=0, exec_async=True)

        out_type = types.FLOAT
        if half:
            out_type = types.FLOAT16

        print('Reading from {}'.format(data_dir))
        file_name = make_dali_cached_file_list_which_is_also_a_file(data_dir)

        self.input = ops.FileReader(
            file_root=data_dir,
            file_list=file_name,
            shard_id=0,
            num_shards=1,
            random_shuffle=True)

        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)

        self.rrc = ops.RandomResizedCrop(device="gpu", size =(crop, crop))

        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=out_type,
            output_layout=types.NCHW,
            crop=(crop, crop),
            image_type=types.RGB,
            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
            std=[0.229 * 255,0.224 * 255,0.225 * 255])

        self.coin = ops.CoinFlip(probability=0.5)
        self.jpegs = None
        self.labels = None

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")

        images = self.decode(self.jpegs)
        images = self.rrc(images)
        output = self.cmnp(images, mirror=rng)

        return [output, self.labels]


class DALISinglePipeAdapter:
    """
        Adapter to make the DALI iterator work as expected
    """
    def __init__(self, dali_iterator):
        self.iterator = dali_iterator

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        # Dali Iterator returns something like [x][y][z]
        #   x - Pipe index (0 in our case)
        #   y - 0 -> data 1 -> label
        #   z -  ??
        val = self.iterator.next()
        return val[0][0][0], val[0][1][0]


def make_dali_loader(args, traindir, crop_size, test_run=True):
    import time

    pipe = HybridTrainPipe(
        batch_size=args.batch_size,
        num_threads=args.workers,
        device_id=0,
        data_dir=traindir,
        crop=crop_size)

    print('Building Pipe')
    pipe.build()

    # No doing the test run just makes it wait somewhere else anyway
    if test_run:
        start = time.time()
        print('Check Pipe')
        pipe.run()
        end = time.time()
        print('Tool {:.4f}s to build pipe'.format(end - start))

    # DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader")))
    return DALISinglePipeAdapter(
        DALIGenericIterator(pipe, ["data", "label"], size=int(pipe.epoch_size("Reader"))))


if __name__ == '__main__':

    pipe = HybridTrainPipe(
        batch_size=256,
        num_threads=8,
        device_id=None,
        data_dir='/home/user1/test_database/train',
        crop=224)

    pipe.build()

    pipe.save_graph_to_dot_file('dali_graph,dot')




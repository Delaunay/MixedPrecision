try:
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=0)

        print('Reading from {}'.format(data_dir))
        self.input = ops.FileReader(
            file_root=data_dir,
            shard_id=0,
            num_shards=1,
            random_shuffle=False)

        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)

        self.rrc = ops.RandomResizedCrop(device="gpu", size =(crop, crop))

        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=types.FLOAT,
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

    def next(self):
        # Dali Iterator returns something like [x][y][z]
        #   x - Pipe index (0 in our case)
        #   y - 0 -> data 1 -> label
        #   z -  ??
        val = self.iterator.next()

        print(val)

        return val[0][0][0], val[0][1][0]


def make_dali_loader(args, traindir, crop_size):
    pipe = HybridTrainPipe(
        batch_size=args.batch_size,
        num_threads=args.workers,
        device_id=0,
        data_dir=traindir,
        crop=crop_size)

    print('Building Pipe')
    pipe.build()

    print('Check Pipe')
    pipe.run()

    print('Data ready!')
    # DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader")))
    return DALISinglePipeAdapter(
        DALIGenericIterator(pipe, ["data", "label"], size=int(pipe.epoch_size("Reader"))))


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=0)
        self.input = ops.FileReader(
            file_root=data_dir, shard_id=1,
            num_shards=1,
            random_shuffle=False)

        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)

        self.res = ops.Resize(device="gpu", resize_shorter=size)

        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=types.FLOAT,
            output_layout=types.NCHW,
            crop=(crop, crop),
            image_type=types.RGB,
            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
            std=[0.229 * 255,0.224 * 255,0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")

        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


import torchvision
import torchvision.transforms as transforms
import time
import torch
import os
import h5py
import numpy as np

from MixedPrecision.tools.stats import StatStream


def preprocess_to_hdf5(transform, input_folder: str, output_file: str):
    train_dataset = torchvision.datasets.ImageFolder(
        input_folder,
        transform)

    output = h5py.File(output_file, 'w', libver='latest')

    # >>>>>>
    # Stores an Array of String representing Index -> class
    classes = output.create_dataset('classes', (1000,), dtype='S9')
    cls = list(train_dataset.class_to_idx.items())
    cls.sort(key=lambda x: x[1])

    for (key, index) in cls:
        classes[index] = np.string_(key)

    # <<<<<<
    n = len(train_dataset)
    hdy = output.create_dataset('label', (n,), dtype=np.uint8)
    hdx = output.create_dataset(
        'data',
        (n, 3, 256, 256),
        dtype=np.uint8,
        chunks=(1, 3, 256, 256),  # Chunk Per sample for fast retrieval
        compression='lzf')

    load_time = StatStream(10)
    save_time = StatStream(10)
    start = time.time()

    print('Converting...')

    for index, (x, y) in enumerate(train_dataset):
        end = time.time()
        load_time += end - start

        s = time.time()

        # convert to uint8
        x = np.array(x, dtype=np.uint8)

        hdy[index] = y
        hdx[index] = np.moveaxis(x, -1, 0)

        e = time.time() 

        save_time += e - s

        if index % 100 == 0 and load_time.avg > 0:
            print('{:.4f} % Load[avg: {:.4f} img/s sd: {:.4f}] Save[avg: {:.4f} img/s sd: {:.4f}]'.format(
                index * 100 / n, 1 / load_time.avg, load_time.sd, 1 / save_time.avg, save_time.sd))

        start = time.time()

    output.close()
    print('{:.4f} img/s'.format(1 / load_time.avg))


class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, file_name: str, transform=None, target_transform=None):
        self.file = h5py.File(file_name, 'r', libver='latest', swmr=True)

        self.transform = transform
        self.target_transform = target_transform

        self.labels = self.file['label']
        self.samples = self.file['data']
        self.size = len(self.file['label'])

    def __getitem__(self, index):
        self.samples.refresh()

        sample = self.samples[index]
        sample = sample.astype(np.uint8)

        if self.transform is not None:
            sample = self.transform(sample)

        target = self.labels[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return self.size

    def __del__(self):
        self.file.close()


def ignore(x, y):
    pass


def main():
    from MixedPrecision.tools.loaders import hdf5_loader
    from MixedPrecision.tools.utils import show_args
    import argparse

    parser = argparse.ArgumentParser('Image Net Preprocessor')
    parser.add_argument('--input', type=str, help='Input directory')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--test-only', action='store_true', default=False,
                        help='Do not run the preprocessor')
    parser.add_argument('--speed-test', action='store_true', default=False,
                        help='Run the speed test on the created dataset')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size to use for the speed test')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of worker to use for the speed trest')

    t = transforms.Compose([
        transforms.Resize((256, 256))
    ])

    args = parser.parse_args()
    show_args(args)

    if not args.test_only:
        s = time.time()
        preprocess_to_hdf5(t, args.input, args.output)
        e = time.time()
        print('Preprocessed Dataset in {:.4f} min'.format((e - s) / 60))

    if args.speed_test:
        print('Speed test')

        # Create a new args that is usable by our data loader
        args = argparse.Namespace(
            data=args.output,
            workers=args.workers,
            batch_size=args.batch_size
        )
        loader = hdf5_loader(args)
        print(' - {} images available'.format(len(loader.dataset)))

        load = StatStream(20)
        start = time.time()

        for index, (x, y) in enumerate(loader):
            end = time.time()
            ignore(x, y)
            load += end - start

            if index > 100:
                break

            start = time.time()

        print(' - {:.4f} img/sec (min={:.4f}, max={:.4f})'.format(
            args.batch_size / load.avg, args.batch_size / load.max, args.batch_size / load.min))

        print(' - {:.4f} sec (min={:.4f}, max={:.4f}, sd={:.4f})'.format(
            load.avg, load.min, load.max, load.sd))


if __name__ == '__main__':
    from MixedPrecision.tools.loaders import hdf5_loader
    from MixedPrecision.tools.utils import show_args
    import argparse

    #main()

    print('Batch Size,	Workers,	Average (s),	SD (s),	Min (s),	Max (s),	Count')
    for w in (0, 1, 2, 4, 8):
        for b in (32, 64, 128, 256):
            # Create a new args that is usable by our data loader
            args = argparse.Namespace(
                data='/home/user1/test_database/imgnet/ImageNet.hdf5',
                workers=w,
                batch_size=b
            )
            loader = hdf5_loader(args)
            load = StatStream(20)
            start = time.time()

            for index, (x, y) in enumerate(loader):
                end = time.time()
                ignore(x, y)
                load += end - start

                if index > 100:
                    break

                start = time.time()

            del loader
            print('{}, {}, {}, {}, {}, {}, {}'.format(b, w, load.avg, load.sd, load.min, load.max, load.count))





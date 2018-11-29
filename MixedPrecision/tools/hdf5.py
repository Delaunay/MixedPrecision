import torchvision
import torchvision.transforms as transforms
import time
import torch
import os
import h5py


from MixedPrecision.tools.stats import StatStream


def preprocess_to_hdf5(transform, input_folder, output_file):
    train_dataset = torchvision.datasets.ImageFolder(
        input_folder,
        transform)

    output = h5py.File(output_file, 'w')

    # >>>>>>
    # Stores an Array of String representing Index -> class
    classes = output.create_dataset('classes', (1000,), dtype='S9')
    cls = list(train_dataset.class_to_idx.items())
    cls.sort(key=lambda x: x[1])

    for (key, index) in cls:
        classes[index] = np.string_(key)
    # <<<<<<

    hdy = output.create_dataset('label', (len(train_dataset),), dtype='i')
    hdx = output.create_dataset('data', (len(train_dataset), 3, 256, 256),
                                dtype='i8', compression='gzip',  compression_opts=4)

    load_time = StatStream(10)
    start = time.time()

    for index, (x, y) in enumerate(train_dataset):
        end = time.time()
        load_time += end - start

        hdy[index] = y
        hdx[index] = np.moveaxis(np.array(x), -1, 0)

        start = time.time()

    print('avg: {:.4f}s sd: {:.4f} {}'.format(load_time.avg, load_time.sd, load_time.count))
    print('{:.4f} img/s'.format(1 / load_time.avg))


class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, file_name, transform=None, target_transform=None):
        self.file = h5py.File(file_name, 'r')
        self.transform = transform
        self.target_transform = target_transform

        self.labels = self.files['label']
        self.samples = self.files['data']
        self.size = len(self.files['label'])

    def __getitem__(self, index):
        sample = self.samples[index]
        if self.transform is not None:
            sample = self.transform(sample)

        target = self.labels[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return self.size


def main():
    import argparse
    parser = argparse.ArgumentParser('Image Net Preprocessor')
    parser.add_argument('--input', type=str, help='Input directory')
    parser.add_argument('--output', type=str, help='Output directory')

    t = transforms.Compose([
        transforms.Resize((256, 256))
        # transforms.RandomResizedCrop(256)
    ])
    args = parser.parse_args()

    preprocess_to_hdf5(t, args.input, args.output)


if __name__ == '__main__':
    import numpy as np
    # Jpegs are split into blocks of 8x8x (8 bit) pixel (512 bits)
    t = transforms.Compose([
        transforms.Resize((256, 256))
        # transforms.RandomResizedCrop(256)
    ])

    preprocess_to_hdf5(t, '/home/user1/test_database/train', '/home/user1/test_database/out.hdf5')

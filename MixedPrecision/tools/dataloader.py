import torchvision.datasets.folder as data
import MixedPrecision.tools.stats as stats
import time


class TimedDatasetFolder:
    """
        A Specialized Version for torchvision datasetFolder that times the time spent transforming the inputs
    """

    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        self.dataset_folder = data.DatasetFolder(root, loader, extensions, transform, target_transform)

        print('Initializing a new TimedDatasetFolder')
        self._read_timer = stats.StatStream(drop_first_obs=10)
        self._transform_timer = stats.StatStream(drop_first_obs=10)

    @property
    def _samples(self):
        return self.dataset_folder.samples

    @property
    def _loader(self):
        return self.dataset_folder.loader

    @property
    def transform(self):
        return self.dataset_folder.transform

    @property
    def target_transform(self):
        return self.dataset_folder.target_transform

    @property
    def read_timer(self):
        return self._read_timer

    @property
    def transform_timer(self):
        return self._transform_timer

    def __getitem__(self, index):
        read_start = time.time()

        path, target = self._samples[index]
        sample = self._loader(path)

        read_end = time.time()
        self._read_timer += (read_end - read_start)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        transform_end = time.time()
        self._transform_timer += (transform_end - read_end)

        return sample, target

    def __len__(self):
        return len(self.dataset_folder)

    def __repr__(self):
        return 'TimedDatasetFolder: ' + repr(self.dataset_folder)


class TimedImageFolder(TimedDatasetFolder):

    def __init__(self, root, transform=None, target_transform=None, loader=data.default_loader):
        super(TimedImageFolder, self).__init__(
            root,
            loader,
            data.IMG_EXTENSIONS,
            transform=transform,
            target_transform=target_transform)

        self._imgs = self._samples


def load_imagenet(folder, transforms):
    import MixedPrecision.tools.utils as utils
    import torch.utils.data

    train_dataset = TimedImageFolder(folder, transforms)

    return torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=None, num_workers=1, pin_memory=True, collate_fn=utils.fast_collate)


if __name__ == '__main__':
    import torchvision.transforms as transforms

    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip()
        # transforms.ToTensor(),
        # normalize,
        # transforms.Lambda(lambda x: utils.enable_cuda(utils.enable_half(x)))
    ])

    data = load_imagenet('/home/user1/test_database/', data_transforms)

    for batch in data:
        print(batch)

    print(data)
    print(data.dataset)
    print(data.dataset.read_timer.to_array())

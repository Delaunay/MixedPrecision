import torchvision.datasets.folder as data
import MixedPrecision.tools.stats as stats
import time


class TimedDatasetFolder(data.DatasetFolder):
    """
        A Specialized Version for torchvision datasetFolder that times the time spent transforming the inputs
    """

    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        super(TimedDatasetFolder, self).__init__(root, loader, extensions, transform, target_transform)
        print('Initializing a new TimedDatasetFolder')
        self.read_timer = stats.StatStream(drop_first_obs=10)
        self.transform_timer = stats.StatStream(drop_first_obs=10)

    def __getitem__(self, index):
        read_start = time.time()

        path, target = self.samples[index]
        sample = self.loader(path)

        read_end = time.time()
        self.read_timer += read_end - read_start

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        transform_end = time.time()
        self.transform_timer = transform_end - read_end

        return sample, target


class TimedImageFolder(TimedDatasetFolder):
    __doc__ = data.ImageFolder.__doc__

    def __init__(self, root, transform=None, target_transform=None,
                 loader=data.default_loader):
        super(TimedDatasetFolder, self).__init__(root, loader, data.IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples

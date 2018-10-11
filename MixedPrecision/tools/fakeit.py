

def fakeit(framework, batch_size, shape, classes, transform=None):
    if framework == 'pytorch':
        return fakeit_pytorch(batch_size, shape, classes, transform)

    return fakeit_tensorflow(batch_size, shape, classes, transform)


def fakeit_pytorch(batch_size, shape, classes, transform):
    from torchvision.datasets.fakedata import FakeData
    from torchvision import transforms

    return FakeData(batch_size, shape, classes,
                    transform=transform,
                    target_transform=transforms.Lambda(lambda x: x.long()))


def fakeit_tensorflow(batch_size, shape, classes, transform):
    pass
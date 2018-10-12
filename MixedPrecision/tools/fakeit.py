

def fakeit(framework, batch_size, shape, classes, transform=None, target_transform=None):
    if framework == 'pytorch':
        return fakeit_pytorch(batch_size, shape, classes, transform, target_transform)

    return fakeit_tensorflow(batch_size, shape, classes, transform, target_transform)


def fakeit_pytorch(batch_size, shape, classes, transform, target_transform):
    from torchvision.datasets.fakedata import FakeData
    from torchvision import transforms

    return FakeData(batch_size, shape, classes,
                    transform=transform,
                    target_transform=target_transform)


def fakeit_tensorflow(batch_size, shape, classes, transform, target_transform):
    pass
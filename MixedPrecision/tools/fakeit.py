
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


def makedirs(dir):
    import os

    try:
        os.makedirs(dir)
    except:
        pass


def make_fake_dataset(shape):
    import torch
    import time
    import torchvision.transforms as transforms

    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).float().view(3, 1, 1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).float().view(3, 1, 1)

    output = '/media/setepenre/UserData/tmp/fake/'
    s = time.time()
    for i in range(0, 256 * 200):
        try:
            target = int(torch.randint(0, 100, (1,)).item())
            eps = torch.randn(shape)
            x = mean + eps * std
            makedirs('{}/{}'.format(output, target))
            img = transforms.ToPILImage()(x)
            file = '{}/{}/{}_{}.jpg'.format(output, target, 'test5', i)
            img.save(file)
        except FileExistsError:
            pass

    print('Time {:.4f}'.format(time.time() - s))



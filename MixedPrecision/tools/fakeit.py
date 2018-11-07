import tarfile
from PIL import Image, ImageFile
from torch.utils.data.dataset import Dataset
import torchvision.datasets.folder


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


def pil_loader(file_object):
    import io
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    bytes = io.BytesIO()
    bytes.write(file_object.read())
    print(bytes)

    img = Image.open(bytes, 'r')  #.load()
    img.load()
    return img.convert('RGB')


class TarDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None,  loader=pil_loader):
        self.tarfile = tarfile.open(root, 'r:*')
        self.loader = loader
        self.x_transform = transform
        self.y_transform = target_transform
        self.classes, self.classes_to_idx, self.files = self.find_classes(self.tarfile.getnames())

    def find_classes(self, files):
        classes = set()
        classes_idx = {}
        nfiles = []

        for file in files:
            try:
                nfiles.append(file)
                _, name, _ = file.split('/')

                if name not in classes:
                    classes_idx[name] = len(classes)
                    classes.add(name)
            except ValueError:
                pass

        return classes, classes_idx, nfiles

    def __getitem__(self, index):
        try:
            import io

            path = self.files[index]
            target = path.split('/')[1]

            file = self.tarfile.extractfile(self.files[index])

            sample = self.loader(file)

            if self.x_transform is not None:
                sample = self.x_transform(sample)
            if self.y_transform is not None:
                target = self.y_transform(target)

            return sample, target
        except OSError as e:
            print('File {} failed to load'.format(self.files[index]))
            raise e

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':

    # make_fake_dataset((3, 1024, 1024))
    data = TarDataset('/media/setepenre/UserData/tmp/fake.tar')




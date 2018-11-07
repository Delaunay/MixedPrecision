import tarfile
import zipfile
from PIL import Image, ImageFile
from torch.utils.data.dataset import Dataset
import torchvision.datasets.folder


def pil_loader(file_object):
    #import io
    #ImageFile.LOAD_TRUNCATED_IMAGES = True
    #bytes = io.BytesIO()
    #bytes.write(file_object.read())

    img = Image.open(file_object, 'r')  #.load()
    return img.convert('RGB')


class TarDataset(Dataset):
    """ Tar files do not work """

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


class ZipDataset(Dataset):
    """ Zip files work """

    def __init__(self, root, transform=None, target_transform=None,  loader=pil_loader):
        self.zipfile = zipfile.ZipFile(root, 'r')
        self.loader = loader
        self.x_transform = transform
        self.y_transform = target_transform
        self.classes, self.classes_to_idx, self.files = self.find_classes(self.zipfile.namelist())

    def find_classes(self, files):
        classes = set()
        classes_idx = {}
        nfiles = []

        for file in files:
            try:
                _, name, file_name = file.split('/')

                if file_name != '':
                    nfiles.append(file)

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
            target = self.classes_to_idx[path.split('/')[1]]

            file = self.zipfile.open(self.files[index], 'r')

            sample = self.loader(file)

            if self.x_transform is not None:
                sample = self.x_transform(sample)
            if self.y_transform is not None:
                target = self.y_transform(target)

            #print(sample, target)
            return sample, target
        except OSError as e:
            print('File {} failed to load'.format(self.files[index]))
            raise e

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':

    # make_fake_dataset((3, 1024, 1024))
    data = ZipDataset('/media/setepenre/UserData/tmp/fake.zip')

    img = data[0]
    print(img)



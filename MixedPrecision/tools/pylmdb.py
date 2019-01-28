import torchvision
import torchvision.transforms as transforms
import time
import torch
import os
import lmdb
import numpy as np

from caffe2.proto import caffe2_pb2
#from caffe2.io import datum_to_array, array_to_datum
import caffe2


from MixedPrecision.tools.stats import StatStream


def preprocess_to_lmdb(transform, input_folder: str, output_file: str):
    train_dataset = torchvision.datasets.ImageFolder(
        input_folder,
        transform)

    n = len(train_dataset)

    print(output_file)
    env = lmdb.open(output_file, map_size=3 * 256 * 256 * n * 9)

    load_time = StatStream(10)
    save_time = StatStream(10)
    start = time.time()

    print('Converting...')

    for index, (x, y) in enumerate(train_dataset):
        end = time.time()
        load_time += end - start
        s = time.time()
        with env.begin(write=True, buffers=True) as txn:
            # convert to uint8
            x = np.array(x, dtype=np.uint8)
            x = np.moveaxis(x, -1, 0)

            #datum = array_to_datum(X, y)

            tensor_protos = caffe2_pb2.TensorProtos()
            img_tensor = tensor_protos.protos.add()
            img_tensor.dims.extend(x.shape)
            img_tensor.data_type = caffe2_pb2.TensorProto.UINT8

            flatten_img = x.reshape(np.prod(x.shape))
            img_tensor.int32_data.extend(flatten_img)

            label_protos = caffe2_pb2.TensorProtos()
            label_tensor = tensor_protos.protos.add()
            label_tensor.data_type = caffe2_pb2.TensorProto.INT32
            label_tensor.int32_data.append(y)

            txn.put('x_{}'.format(index).encode('ascii'), tensor_protos.SerializeToString())
            txn.put('y_{}'.format(index).encode('ascii'), label_protos.SerializeToString())

        e = time.time()

        save_time += e - s

        if index % 100 == 0 and load_time.avg > 0:
            print('{:.4f} % Load[avg: {:.4f} img/s sd: {:.4f}] Save[avg: {:.4f} img/s sd: {:.4f}]'.format(
                index * 100 / n, 1 / load_time.avg, load_time.sd, 1 / save_time.avg, save_time.sd))

        start = time.time()

    env.close()
    print('{:.4f} img/s'.format(1 / load_time.avg))


class LMDBDataset(torch.utils.data.Dataset):
    def __init__(self, file_name: str, transform=None, target_transform=None):
        self.env = lmdb.open(file_name, readonly=True)
        self.target_transform = target_transform
        self.transform = transform

    def __getitem__(self, index):
        datum = caffe2_pb2.Datum()

        with self.env.begin() as txn:
            key = '{}'.format(index).encode('ascii')
            proto = txn.get(key)
            datum.ParseFromString(proto)

            label = datum.label
            data = caffe2.io.datum_to_array(datum).astype(np.uint8)

            print(label)
            print(data)

        sample = sample.astype(np.uint8)

        if self.transform is not None:
            sample = self.transform(sample)

        target = self.labels[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return self.size


if __name__ == '__main__':
    # --input /home/user1/test_database/train/ --output /home/user1/test_database/imgnet/ImageNet.hdf5 --speed-test --workers 8 --batch-size 256

    t = transforms.Compose([
        transforms.Resize((256, 256))
    ])

    # preprocess_to_lmdb(t, '/home/user1/test_database/train/', '/home/user1/test_database/imgnet/ImageNet.lmdb')

    db = LMDBDataset('/home/user1/test_database/imgnet/ImageNet.lmdb')

    print(db[0])




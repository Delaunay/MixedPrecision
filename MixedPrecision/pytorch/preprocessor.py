import torchvision
import torchvision.transforms as transforms
import time
import os

from MixedPrecision.tools.stats import StatStream


def preprocess(transform, input_folder, output_folder):
    train_dataset = torchvision.datasets.ImageFolder(
        input_folder,
        transform)

    load_time = StatStream(10)

    start = time.time()
    for index, (x, y) in enumerate(train_dataset):
        end = time.time()
        load_time += end - start

        class_name = train_dataset.classes[y]
        output_dir = '{}/{}'.format(output_folder, class_name)
        os.makedirs(output_dir, mode=0o755, exist_ok=True)

        out = '{}/{}_{}.jpeg'.format(output_dir, class_name, class_name, index)
        x.save(out, 'JPEG')

        start = time.time()

    print('avg: {:.4f}s sd: {:.4f} {}'.format(load_time.avg, load_time.sd, load_time.count))
    print('{:.4f} img/s'.format(1 / load_time.avg))


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

    preprocess(t, args.input, args.output)


if __name__ == '__main__':
    # Jpegs are split into blocks of 8x8x (8 bit) pixel (512 bits)
    t = transforms.Compose([
        transforms.Resize((256, 256))
        # transforms.RandomResizedCrop(256)
    ])

    preprocess(t, '/home/user1/test_database/train', '/home/user1/test_database/out')


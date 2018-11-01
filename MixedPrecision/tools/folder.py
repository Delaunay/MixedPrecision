from typing import *
import os
import fnmatch
import time
import hashlib


def pytorch_folder_visitor(folder):
    images = []
    classes = set()

    for root, dir, files in os.walk(folder):
        classes.add(root.split('/')[-1])
        for item in fnmatch.filter(files, "*"):
            images.append(root + '/' + item)

    return images, {k: i for i, k in enumerate(classes)}


def dali_folder_visitor(folder) -> List[str]:
    classes = set()

    for root, _, files in os.walk(folder):
        if root != folder:
            classes.add(root.split('/')[-1])

        for item in fnmatch.filter(files, "*"):
            yield '{} {}'.format(item, len(classes) - 1)


def make_dali_cached_file_list_which_is_also_a_file(data_dir):
        start = time.time()
        h = hashlib.sha256()
        h.update(data_dir.encode('utf-8'))
        file_name = 'tmp_' + h.hexdigest()

        if not os.path.isfile(file_name):
            images = '\n'.join(dali_folder_visitor(data_dir))
            file = open(file_name, 'w')
            file.write(images)
            file.close()

        end = time.time()
        print('Took {} to walk folder'.format(end - start))
        return file_name


if __name__ == '__main__':
    img = make_dali_cached_file_list_which_is_also_a_file('/home/user1/test_database/train')
    print(img)

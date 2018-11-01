from typing import *
import os
import fnmatch
import time
import hashlib
import tempfile


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
        name = root.split('/')[-1]
        if root != folder:
            classes.add(name)

        for item in fnmatch.filter(files, "*"):
            yield '{} {}'.format(name + '/' + item, len(classes) - 1)


def make_dali_cached_file_list_which_is_also_a_file(data_dir):
    # I want to make a temp file that is going to be persistent for `some` time
    # so the file name should be consistent and unique per folder

    h = hashlib.sha256()
    h.update(data_dir.encode('utf-8'))
    temp_dir = tempfile.gettempdir()
    file_name = temp_dir + '/tmp_image_db_dali_' + h.hexdigest()

    if not os.path.isfile(file_name):
        start = time.time()

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

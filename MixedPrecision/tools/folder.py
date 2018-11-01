from typing import *
import os
import fnmatch


def pytorch_folder_visitor(folder):
    images = []
    classes = set()

    for root, dir, files in os.walk(folder):
        classes.add(root.split('/')[-1])
        for item in fnmatch.filter(files, "*"):
            images.append(root + '/' + item)

    return images, {k: i for i, k in enumerate(classes)}


def dali_folder_visitor(folder) -> List[Tuple[str, int]]:
    images = []
    classes = set()

    for root, dir, files in os.walk(folder):
        if root != folder:
            classes.add(root.split('/')[-1])

        for item in fnmatch.filter(files, "*"):
            yield (root + '/' + item, len(classes) - 1)


if __name__ == '__main__':

    img = dali_folder_visitor('/home/user1/test_database/train')
    print(list(img))

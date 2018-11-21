#code borrowed from https://github.com/bohanwu/cs231n-final-project/blob/master/data/princeton_sunrgbd/load_data.py

import argparse
import os
import sys

import numpy as np
from PIL import Image


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_set_dir', type=str,
                        help='Directory of the test dataset.',
                        default='./SUNRGBDv2Test')
    parser.add_argument('--training_set_dir', type=str,
                        help='Directory of the training dataset.',
                        default='./SUNRGBD')
    return parser.parse_args(argv)


def get_rgbd_np_array(dir):
    desired_dimension = (640, 480)
    print(dir)
    image_dir = os.path.join(dir, "image")
    for i in os.listdir(image_dir):
        if ".jpg" in i:
            if "resized" not in i:
                # original image
                if "resized_%s" % i not in os.listdir(image_dir):
                    # resize and save image
                    print("Creating resized color image: resized_%s" % i)
                    im = Image.open(os.path.join(image_dir, i))
                    out = im.resize(desired_dimension, Image.ANTIALIAS)
                    im = Image.fromarray(np.array(out))
                    im.save(os.path.join(image_dir, "resized_%s" % i))

                im = Image.open(os.path.join(image_dir, "resized_%s" % i))
                image = np.array(im)
        else:
            if i != ".DS_Store":
                print("something is wrong: %s" % os.path.join(image_dir, i))

    depth_dir = os.path.join(dir, "depth")
    for i in os.listdir(depth_dir):
        if ".png" in i:
            if "resized" not in i:
                # original depth image
                if "resized_%s" % i not in os.listdir(depth_dir):
                    # resize and save depth image
                    print("Creating resized depth image: resized_%s" % i)
                    im = Image.open(os.path.join(depth_dir, i))
                    out = im.resize(desired_dimension, Image.ANTIALIAS)
                    im = Image.fromarray(np.array(out))
                    im.save(os.path.join(depth_dir, "resized_%s" % i))

                im = Image.open(os.path.join(depth_dir, "resized_%s" % i))
                depth = np.array(im)
        else:
            if i != ".DS_Store":
                print("something is wrong: %s" % os.path.join(image_dir, i))

    return np.concatenate((image, np.reshape(depth, (
        np.shape(depth)[0], np.shape(depth)[1], 1))), axis=2)


def get_label(dir):
    label_dir = os.path.join(dir, "scene.txt")
    with open(label_dir, 'r') as f:
        return f.read()


# X_test is 2860 x 480 x 640 x 4. 2860 is num test examples, 640 is image
# width, 480 is image height, 4 is all channels with the last being depth.
def get_rgbd_test_set(dir):
    X_test = np.zeros((2860, 480, 640, 4))
    count = 0
    dirs = ["depth", "image"]
    num_dirs = len(dirs)
    root_dir = os.path.expanduser(dir)
    for sub_dir in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, sub_dir)):
            for sub_dir_2 in os.listdir(os.path.join(root_dir, sub_dir)):
                full_dir = os.path.join(
                    os.path.join(root_dir, sub_dir, sub_dir_2))
                if os.path.isdir(full_dir):
                    categories = os.listdir(full_dir)
                    if sum([int(i in categories) for i in dirs]) == num_dirs:
                        X_test[count, :, :, :] = get_rgbd_np_array(full_dir)
                        count += 1

    return X_test


# X_train is 10,355 x 480 x 640 x 4. 10,355 is num test examples, 640 is image
# width, 480 is image height, 4 is all channels with the last being depth.
# The original images given were not the same in size, so we reshape them
# into a standard 480 height x 640 width dimension.

# y_train is 10,355 english scene names
def get_rgbd_training_set(dir):
    def recursive_search(dir):
        dir_lists = os.listdir(dir)
        if not "intrinsics.txt" in dir_lists:
            for i in dir_lists:
                sub_dir = os.path.join(dir, i)
                if os.path.isdir(sub_dir):
                    recursive_search(sub_dir)
        else:
            if sum([int(i in dir_lists) for i in dirs]) == num_dirs:
                X_train[counts[0], :, :, :] = get_rgbd_np_array(dir)
                y_train.append(get_label(dir))
                counts[0] += 1

    X_train, y_train = np.zeros((10355, 480, 640, 4)), []
    counts = [0]
    dirs = ["depth", "image", "scene.txt"]
    num_dirs = len(dirs)

    recursive_search(os.path.expanduser(dir))
    print(counts[0])
    return X_train, y_train


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    # X_test = get_rgbd_test_set(args.test_set_dir)
    X_train, y_train = get_rgbd_training_set(args.training_set_dir)
    print(X_train.shape, y_train.shape)
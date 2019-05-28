from scipy import io as sio
import cv2
import os
import numpy as np
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
import argparse


def get_annotations(file_name: str)->list:
    mat = sio.loadmat(file_name)
    return mat['annotations']


def get_label_info(contents: list)->list:
    label_info = []

    for c in contents[0]:
        label_info.append({
            'x1': int(c[0][0]),
            'y1': int(c[1][0]),
            'x2': int(c[2][0]),
            'y2': int(c[3][0]),
            'class': int(c[4][0]),
            'file_name': c[5][0]
        })

    return label_info


def split_image(images_names, label_info, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    total_len = len(images_names)
    count = 0
    for image_name, label in tqdm(zip(images_names, label_info), total=total_len, ascii=True):
        count += 1
        image = cv2.imread(image_name)

        new_image = image[label['y1']:label['y2'], label['x1']:label['x2']]
        new_path = os.path.join(save_dir, label['file_name'])
        cv2.imwrite(new_path, new_image)
        label['file_name'] = new_path

    print("{}/{} is Split".format(count, total_len))


def save_label_info(label_info, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(label_info, f)


def expand(label_info, save_dir, mode='lr'):

    print("Mode: ", mode)

    new_info = label_info.copy()
    images_len = len(label_info)

    print(images_len)

    next_index = images_len+1

    for label in tqdm(label_info, total=images_len, ascii=True):
        image = cv2.imread(label['file_name'])

        if mode == 'lr':
            new_image = cv2.flip(image, 1)
        elif mode == 'RGB':
            new_image = image[:, :, ::-1]

        new_image_name = str(next_index).zfill(5)+'.jpg'
        new_image_path = os.path.join(save_dir, new_image_name)
        cv2.imwrite(new_image_path, new_image)
        next_index += 1

        new_info.append({
            'x1': label['x1'],
            'y1': label['y1'],
            'x2': label['x2'],
            'y2': label['y2'],
            'class': label['class'],
            'file_name': new_image_path
        })

    return new_info


def run(mode, args):
    old_img_dir = "./data/cars_train/" if mode == 'train' else './data/cars_test'
    annos_dir = './data/devkit/cars_train_annos.mat' if mode == 'train' else './data/devkit/cars_test_annos_withlabels.mat'
    new_img_dir = './data/new_cars_train' if mode == 'train' else './data/new_cars_test'
    label_info_file_name = os.path.join(new_img_dir, mode+".pickle")

    print("Get {} annotations".format(mode))
    annotations = get_annotations(annos_dir)

    print("Get {} label".format(mode))
    label_info = get_label_info(annotations)

    images_names = [os.path.join(old_img_dir, dir_)
                    for dir_ in sorted(os.listdir(old_img_dir))]

    print("Start split {} image".format(mode))
    split_image(images_names, label_info, new_img_dir)

    if args.expand_left_right and mode == 'train':
        print("expand {}  left and right ........".format(mode))
        label_info = expand(label_info, new_img_dir, 'lr')

    if args.expand_rgb and mode == 'train':
        print("expand {}  RGB ........".format(mode))
        label_info = expand(label_info, new_img_dir, 'RGB')

    print("Total images: ", len(label_info))
    save_label_info(label_info, label_info_file_name)


def str2bool(s):
    if s in ('true', 'True'):
        return True
    elif s in ('false', "False"):
        return False
    else:
        raise argparse.ArgumentTypeError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--expand_left_right', type=str2bool, default=False)
    parser.add_argument('--expand_rgb', type=str2bool, default=False)
    args = parser.parse_args()

    print("Start Train images.......")
    run('train', args)

    print("Start Test images.......")
    run('test', args)

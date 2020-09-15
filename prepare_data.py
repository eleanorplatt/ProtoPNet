import os
import numpy as np # linear algebra
import xml.etree.ElementTree as ET # for parsing XML
import matplotlib.pyplot as plt # to show images
from PIL import Image

root_images = "../data/CUB_200_2011/CUB_200_2011/images"
root_images_list = "../data/CUB_200_2011/CUB_200_2011/images.txt"
root_bounding_boxes = "../data/CUB_200_2011/CUB_200_2011/bounding_boxes.txt"
root_split = "../data/CUB_200_2011/CUB_200_2011/train_test_split.txt"
root_target = "../data/cub200_cropped"

all_classes = os.listdir(root_images)
print("Number of classes: {}".format(len(all_classes)))

# create new directory and subdirectories for cropped images
if not os.path.exists(root_target):
    os.makedirs(root_target)
if not os.path.exists(os.path.join(root_target, 'train_cropped')):
    os.makedirs(os.path.join(root_target, 'train_cropped'))
if not os.path.exists(os.path.join(root_target, 'test_cropped')):
    os.makedirs(os.path.join(root_target, 'test_cropped'))
for c in all_classes:
    subdir_train = os.path.join(root_target, 'train_cropped', c)
    if not os.path.exists(subdir_train):
        os.makedirs(subdir_train)
    subdir_test = os.path.join(root_target, 'test_cropped', c)
    if not os.path.exists(subdir_test):
        os.makedirs(subdir_test)

# load text files
with open(root_images_list) as f:
    images_list = f.read().splitlines() 
with open(root_bounding_boxes) as f:
    bb_list = f.read().splitlines() 
with open(root_split) as f:
    split_list = f.read().splitlines() 
assert(len(bb_list) == len(images_list))
assert(len(split_list) == len(images_list))

for i, img in enumerate(images_list):

    # crop image
    img_idx, path = img.split(' ')
    bb = bb_list[i]
    bb_idx, x, y, w, h = bb.split(' ')
    assert(img_idx == bb_idx)
    left = int(float(x))
    top = int(float(y))
    right = int(float(x)) + int(float(w))
    bottom = int(float(y)) + int(float(h))
    im = Image.open(os.path.join(root_images,path))
    im = im.crop((left, top, right, bottom))

    # save image
    s = split_list[i]
    split_idx, train = s.split(' ')
    train = int(train)
    assert(split_idx == img_idx)
    if train==1:
        path_to_save = os.path.join(root_target, 'train_cropped', path)
    elif train==0:
        path_to_save = os.path.join(root_target, 'test_cropped', path)
    im.save(path_to_save, "JPEG")
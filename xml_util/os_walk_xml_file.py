import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, dump, ElementTree
import shutil


src_dir_folder = ['train', 'test']
src_dir = 'C:/Users/chzhq/Desktop/checked_image/R2CNN/goodroof_parkinglot_solarpanel_rooftop_facility_tight_heliport/tmp/'
separated_dir = 'C:/Users/chzhq/Desktop/checked_image/R2CNN/goodroof_parkinglot_solarpanel_rooftop_facility_tight_heliport/dst'

src_anno_dir_list = []
dst_anno_dir_list = []
src_img_dir_list = []
dst_img_dir_list = []
for folder in src_dir_folder:
    src_anno_dir_list.append(os.path.join(src_dir, folder, 'annotations'))
    dst_anno_dir_list.append(os.path.join(separated_dir, folder, 'annotations'))
    src_img_dir_list.append(os.path.join(src_dir, folder, 'JPEGImages'))
    dst_img_dir_list.append(os.path.join(separated_dir, folder, 'JPEGImages'))
print(src_anno_dir_list)


def is_exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


for src_dir in src_anno_dir_list:
    index = src_anno_dir_list.index(src_dir)
    # is_exist_dir(dst_img_dir_list[index])
    # is_exist_dir(dst_anno_dir_list[index])
    for root, dirs, files in os.walk(src_dir):
        print('srcdir:', src_dir, " index : ", index)
        print('dstdir:', dst_anno_dir_list[index])
        for file in files:
            current_anno_file = os.path.join(str(root), file)
            current_img_file = os.path.join(src_img_dir_list[index], file[:-3] + 'png')

            dst_anno_file = os.path.join(dst_anno_dir_list[index], file)
            dst_img_file = os.path.join(dst_img_dir_list[index], file[:-3] + 'png')

            # To Do
            print(current_img_file)
            print(current_anno_file)
            print(dst_img_file)
            print(dst_anno_file)
            # To DO

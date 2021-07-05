import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, dump, ElementTree
import shutil


src_dir_folder = ['train', 'test']
src_dir = '/home/qisens-18045/work/dataset/'
separated_dir = '/home/qisens-18045/work/dataset_renamed/'

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

err_file_cnt = 0
for src_dir in src_anno_dir_list:
    index = src_anno_dir_list.index(src_dir)
    is_exist_dir(dst_img_dir_list[index])
    is_exist_dir(dst_anno_dir_list[index])
    for root, dirs, files in os.walk(src_dir):
        print('srcdir:', src_dir, " index : ", index)
        print('dstdir:', dst_anno_dir_list[index])
        for file in files:
            current_anno_file = os.path.join(str(root), file)
            current_img_file = os.path.join(src_img_dir_list[index], file[:-3] + 'png')

            dst_anno_file = os.path.join(dst_anno_dir_list[index], file)
            dst_img_file = os.path.join(dst_img_dir_list[index], file[:-3] + 'png')

            # To Do
            try:
                xml_file = current_anno_file
                xml_tree = ET.parse(xml_file)

                for object in xml_tree.iter("object"):
                    # label = object.find('name').text
                    if object.find('name').text == 'solarpanel-flat':
                        object.find('name').text = 'solarpanel_flat'
                        label = object.find('name').text

                    if object.find('name').text == 'solarpanel-slope':
                        object.find('name').text = 'solarpanel_slope'
                        label = object.find('name').text

                    if object.find('name').text == 'goodroof':
                        object.find('name').text = 'flatroof'
                        label = object.find('name').text

                new_tree = xml_tree
                new_tree.write(dst_anno_file)
                shutil.copyfile(current_img_file, dst_img_file)


                # bndbox_name_elements = xml_tree.findall('*/bndbox')  # bndbox 태그를 탐색
                # if not len(bndbox_name_elements) == 0:  # bndbox 가 있는 경우
                #     print(bndbox_name_elements)

                # robndbox_name_elements = xml_tree.findall('*/robndbox')  # robndbox 태그를 탐색
                # if not len(robndbox_name_elements) == 0:  # robndbox 가 있는 경우
                #     print(robndbox_name_elements)

                # shutil.copyfile(current_anno_file, dst_anno_file)
                # shutil.copyfile(current_img_file, dst_img_file)


            except FileNotFoundError:
                err_file_cnt += 1
                continue
            # To DO
print("Error : ", err_file_cnt)

import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, dump, ElementTree
import shutil

# 이 리스트 안에 해당하는 클래스가 있다면 이미지와 annotaion 파일을 분리(annotation 파일 이름 기준으로 이미지를 찾음)
# dir - images
#     ㄴ annotaions    이 구조여야 함
label_list = ['solarpanel']
# label_list = ['solarpanel']

# src_dir_folder = ['train', 'test', 'coa_origin']
src_dir_folder = ['train', 'test', 'coa_origin']
src_dir = '/home/qisens/Desktop/separated/rebuilt/'
separated_dir = '/home/qisens/Desktop/separated/filtered/'

error_file = open("../error.txt", "w")

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


cnt = 0
for src_dir in src_anno_dir_list:
    index = src_anno_dir_list.index(src_dir)
    is_exist_dir(dst_img_dir_list[index])
    is_exist_dir(dst_anno_dir_list[index])
    for root, dirs, files in os.walk(src_dir):
        print('srcdir:', src_dir, " index : ", index)
        print('dstdir:', dst_anno_dir_list[index])
        for file in files:
            try:
                current_anno_file = os.path.join(str(root), file)
                current_img_file = os.path.join(src_img_dir_list[index], file[:-3] + 'png')
                xml_file = current_anno_file
                xml_trees = ET.parse(xml_file)

                #root 노드 가져오기
                xml_root = xml_trees.getroot()

                obj_tag = xml_root.findall("object")
                for object in xml_root.iter("object"):
                    label = object.findtext('name')
                    if label in label_list:
                        dst_anno_file = os.path.join(dst_anno_dir_list[index], file)
                        dst_img_file = os.path.join(dst_img_dir_list[index], file[:-3] + 'png')
                        print(dst_anno_file)
                        print("이새끼 범인이야")
                        os.remove(current_img_file)
                        os.remove(current_anno_file)
                        # shutil.copyfile(current_anno_file, dst_anno_file)
                        # shutil.copyfile(current_img_file, dst_img_file)
                        # shutil.move(current_anno_file, dst_anno_file)
                        # shutil.move(current_img_file, dst_img_file)
                        print(label)
                        cnt += 1

            except FileNotFoundError:
                current_anno_file = os.path.join(str(root), file)
                current_img_file = os.path.join(src_img_dir_list[index], file[:-3] + 'png')
                print(current_anno_file)
                print(current_img_file)

                error_file.write(str(file) + '\n')
error_file.close()
print(cnt)

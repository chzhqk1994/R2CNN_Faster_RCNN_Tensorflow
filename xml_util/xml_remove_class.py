import os
import xml.etree.ElementTree as ET
import shutil


label_list = ['facility']  # will be remove
src_dir_folder = ['new_area2']
src_dir = '/media/qisens/4tb3/kowa_global/r2cnn_dataset/flatroof_parkinglot_solarpanel_slope_flat_rooftop_facility_tight_heliport/origin_before_rebuild_saprate_area2_400/tmp/'
separated_dir = '/media/qisens/4tb3/kowa_global/r2cnn_dataset/flatroof_parkinglot_solarpanel_slope_flat_rooftop_facility_tight_heliport/origin_before_rebuild_saprate_area2_400/tmp/facility_removed_400/'

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


def check_class_and_remove(current_xml_file):
    xml_trees = ET.parse(current_xml_file)

    # root 노드 가져오기
    xml_root = xml_trees.getroot()

    for object in xml_trees.findall("object"):
        label = object.find("name").text
        if label in label_list:
            xml_root.remove(object)

    # label 파일에 object 가 없으면 None 을 리턴
    object_list = []
    for object in xml_trees.findall("object"):
        object_list.append(object)
    if len(object_list) == 0:
        return None

    return xml_trees


for src_dir in src_anno_dir_list:
    index = src_anno_dir_list.index(src_dir)
    err_file_cnt = 0
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

            print(current_img_file)
            print(current_anno_file)

            # To Do
            try:

                new_xml_tree = check_class_and_remove(current_anno_file)

                if new_xml_tree is None:
                    shutil.copyfile(current_img_file, dst_img_file)
                    continue

                new_xml_tree.write(dst_anno_file)
                shutil.copyfile(current_img_file, dst_img_file)

            except FileNotFoundError:
                print("ERROR : ", file)
                err_file_cnt += 1
                continue
    print("Error : ", err_file_cnt)
    # To DO

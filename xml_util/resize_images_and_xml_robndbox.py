import os
import math
import cv2
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree
from xml.dom import minidom
import shutil
import numpy as np


def is_exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="        ")


def make_dir_list(src_dir_folder, src_dir, separated_dir):
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

    return src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list


def rotatePoint(xc, yc, xp, yp, theta):
    xoff = xp - xc
    yoff = yp - yc

    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    pResx = cosTheta * xoff + sinTheta * yoff
    pResy = - sinTheta * xoff + cosTheta * yoff
    return str(xc + pResx), str(yc + pResy)


def build_xml_arch(parsed_xml, new_height, new_width, origin_height, origin_width, resize_ratio):
    label = ''
    cx = 0
    cy = 0
    width = 0
    height = 0
    angle = 0
    x0 = 0
    x1 = 0
    x2 = 0
    x3 = 0
    y0 = 0
    y1 = 0
    y2 = 0
    y3 = 0
    result = None

    annotation = Element('annotation')

    folder = SubElement(annotation, 'folder')
    folder.text = parsed_xml['folder']

    filename = SubElement(annotation, 'filename')
    filename.text = parsed_xml['filename']

    path = SubElement(annotation, 'path')
    path.text = parsed_xml['path']

    source = SubElement(annotation, 'source')
    database = SubElement(source, 'database')
    database.text = 'Unknown'

    size = SubElement(annotation, 'size')
    width = SubElement(size, 'width')
    width.text = str(origin_width)
    height = SubElement(size, 'height')
    height.text = str(origin_height)
    depth = SubElement(size, 'depth')
    depth.text = str(3)

    segmented = SubElement(annotation, 'segmented')
    segmented.text = '0'

    for obj in parsed_xml['objects']:
        # xml 태그가 robndbox 인 경우
        if not obj.find('robndbox') is None:
            if resize_ratio > 1:
                horizontal_extra = int((new_width - origin_width) / 2)
                vertical_extra = int((new_height - origin_height) / 2)

                try:
                    label = obj.find('name').text
                    cx = str(float(obj.find('./robndbox/cx').text) * resize_ratio - horizontal_extra)
                    cy = str(float(obj.find('./robndbox/cy').text) * resize_ratio - vertical_extra)
                    width = str(float(obj.find('./robndbox/w').text) * resize_ratio)
                    height = str(float(obj.find('./robndbox/h').text) * resize_ratio)
                    angle = obj.find('./robndbox/angle').text

                    if float(cx) < 0 or float(cx) > origin_width:
                        continue
                    if float(cy) < 0 or float(cy) > origin_height:
                        continue

                except TypeError:
                    continue

                try:
                    x0 = str(float(obj.find('./robndbox/x0').text) * resize_ratio - horizontal_extra)
                    x1 = str(float(obj.find('./robndbox/x1').text) * resize_ratio - horizontal_extra)
                    x2 = str(float(obj.find('./robndbox/x2').text) * resize_ratio - horizontal_extra)
                    x3 = str(float(obj.find('./robndbox/x3').text) * resize_ratio - horizontal_extra)
                    y0 = str(float(obj.find('./robndbox/y0').text) * resize_ratio - vertical_extra)
                    y1 = str(float(obj.find('./robndbox/y1').text) * resize_ratio - vertical_extra)
                    y2 = str(float(obj.find('./robndbox/y2').text) * resize_ratio - vertical_extra)
                    y3 = str(float(obj.find('./robndbox/y3').text) * resize_ratio - vertical_extra)
                except AttributeError:
                    print("누락")
                    cx = float(cx)
                    cy = float(cy)
                    w = float(width)
                    h = float(height)
                    angle = float(angle)

                    x0, y0 = rotatePoint(cx, cy, cx - w / 2, cy - h / 2, -angle)
                    x1, y1 = rotatePoint(cx, cy, cx + w / 2, cy - h / 2, -angle)
                    x2, y2 = rotatePoint(cx, cy, cx + w / 2, cy + h / 2, -angle)
                    x3, y3 = rotatePoint(cx, cy, cx - w / 2, cy + h / 2, -angle)

                    cx, cy, w, h, angle = str(cx), str(cy), str(w), str(h), str(angle)

                    # str() >> 좌표들을 xml로 저장하기 위해선 text 로 바꿔야 하기 때문에 str 로 묶음
                    # int() >> 좌표를 R2CNN 에선 정수형으로 변환하기 때문에 int 로 묶음
                    # float() >> 실수형 좌표를 정수로 바꾸기 위해 float 으로 먼저 형변환을 해줌 >> 왜 해야하는거지
                    x0, x1, x2, x3, y0, y1, y2, y3 = str(int(float(x0))), str(int(float(x1))), str(int(float(x2))), str(
                        int(float(x3))), str(int(float(y0))), str(int(float(y1))), str(int(float(y2))), str(
                        int(float(y3)))

            # 1보다 작으면 모든 bbox 가 포함되니 noise 가 포함된 크기만큼 좌표가 이동할것
            else:
                horizontal_padding = int((origin_width - new_width) / 2)
                vertical_padding = int((origin_height - new_height) / 2)

                try:
                    label = obj.find('name').text
                    cx = str(float(obj.find('./robndbox/cx').text) * resize_ratio + horizontal_padding)
                    cy = str(float(obj.find('./robndbox/cy').text) * resize_ratio + vertical_padding)
                    width = str(float(obj.find('./robndbox/w').text) * resize_ratio)
                    height = str(float(obj.find('./robndbox/h').text) * resize_ratio)
                    angle = obj.find('./robndbox/angle').text

                except TypeError:
                    continue

                try:
                    x0 = str(float(obj.find('./robndbox/x0').text) * resize_ratio + horizontal_padding)
                    x1 = str(float(obj.find('./robndbox/x1').text) * resize_ratio + horizontal_padding)
                    x2 = str(float(obj.find('./robndbox/x2').text) * resize_ratio + horizontal_padding)
                    x3 = str(float(obj.find('./robndbox/x3').text) * resize_ratio + horizontal_padding)
                    y0 = str(float(obj.find('./robndbox/y0').text) * resize_ratio + vertical_padding)
                    y1 = str(float(obj.find('./robndbox/y1').text) * resize_ratio + vertical_padding)
                    y2 = str(float(obj.find('./robndbox/y2').text) * resize_ratio + vertical_padding)
                    y3 = str(float(obj.find('./robndbox/y3').text) * resize_ratio + vertical_padding)
                except AttributeError:
                    print("누락")
                    cx = float(cx)
                    cy = float(cy)
                    w = float(width)
                    h = float(height)
                    angle = float(angle)

                    x0, y0 = rotatePoint(cx, cy, cx - w / 2, cy - h / 2, -angle)
                    x1, y1 = rotatePoint(cx, cy, cx + w / 2, cy - h / 2, -angle)
                    x2, y2 = rotatePoint(cx, cy, cx + w / 2, cy + h / 2, -angle)
                    x3, y3 = rotatePoint(cx, cy, cx - w / 2, cy + h / 2, -angle)

                    cx, cy, w, h, angle = str(cx), str(cy), str(w), str(h), str(angle)

                    # str() >> 좌표들을 xml로 저장하기 위해선 text 로 바꿔야 하기 때문에 str 로 묶음
                    # int() >> 좌표를 R2CNN 에선 정수형으로 변환하기 때문에 int 로 묶음
                    # float() >> 실수형 좌표를 정수로 바꾸기 위해 float 으로 먼저 형변환을 해줌 >> 왜 해야하는거지
                    x0, x1, x2, x3, y0, y1, y2, y3 = str(int(float(x0))), str(int(float(x1))), str(int(float(x2))), str(
                        int(float(x3))), str(int(float(y0))), str(int(float(y1))), str(int(float(y2))), str(int(float(y3)))

            object_tag = SubElement(annotation, 'object')
            type = SubElement(object_tag, 'type')
            type.text = 'robndbox'
            name = SubElement(object_tag, 'name')
            name.text = label
            pose = SubElement(object_tag, 'pose')
            pose.text = 'Unspecified'
            truncated = SubElement(object_tag, 'truncated')
            truncated.text = '0'
            difficult = SubElement(object_tag, 'difficult')
            difficult.text = '0'
            robndbox = SubElement(object_tag, 'robndbox')
            cx_tag = SubElement(robndbox, 'cx')
            cx_tag.text = cx
            cy_tag = SubElement(robndbox, 'cy')
            cy_tag.text = cy
            width_tag = SubElement(robndbox, 'w')
            width_tag.text = width
            height_tag = SubElement(robndbox, 'h')
            height_tag.text = height
            angle_tag = SubElement(robndbox, 'angle')
            angle_tag.text = angle
            x0_tag = SubElement(robndbox, 'x0')
            x0_tag.text = x0
            x1_tag = SubElement(robndbox, 'x1')
            x1_tag.text = x1
            x2_tag = SubElement(robndbox, 'x2')
            x2_tag.text = x2
            x3_tag = SubElement(robndbox, 'x3')
            x3_tag.text = x3
            y0_tag = SubElement(robndbox, 'y0')
            y0_tag.text = y0
            y1_tag = SubElement(robndbox, 'y1')
            y1_tag.text = y1
            y2_tag = SubElement(robndbox, 'y2')
            y2_tag.text = y2
            y3_tag = SubElement(robndbox, 'y3')
            y3_tag.text = y3

        # xml 태그가 bndbox 인 경우
        if not obj.find('bndbox') is None:
            continue

    result = ElementTree(annotation)
    # result = prettify(annotation)
    # print(result)
    # print('\n\n\n')
    return result


def make_noise(resized_image, resized_width, resized_height, origin_height, origin_width):
    noise_img = (np.uint8(np.random.rand(origin_width, origin_height, 3) * 255))

    horizontal_padding = int((origin_width - resized_width) / 2)
    vertical_padding = int((origin_height - resized_height) / 2)
    noise_img[horizontal_padding:horizontal_padding + resized_width,
              vertical_padding:vertical_padding + resized_height] = resized_image
    return noise_img


def xml_parser(root, file, new_height, new_width, origin_height, origin_width, resize_ratio):
    xml_parse = {'folder': '', 'filename': '', 'path': '', 'objects': []}

    file_path = os.path.join(root, file)
    tree = ET.parse(file_path)

    obj_name_elements = tree.findall('./object')
    if len(obj_name_elements) == 0:  # object 가 하나도 없다면 return None
        return None
    xml_parse['objects'] = obj_name_elements

    bndbox_name_elements = tree.findall('*/bndbox')
    if len(bndbox_name_elements) == 0:
        pass

    robndbox_name_elements = tree.findall('*/robndbox')
    if len(robndbox_name_elements) == 0:
        pass

    xml_parse['folder'] = tree.getroot().find('folder').text
    xml_parse['filename'] = tree.getroot().find('filename').text
    xml_parse['path'] = tree.getroot().find('path').text

    new_xml_tree = build_xml_arch(xml_parse, new_height, new_width, origin_height, origin_width, resize_ratio)

    new_obj_name_elements = new_xml_tree.findall('./object')
    if len(new_obj_name_elements) == 0:  # object 가 하나도 없다면 return None
        return None

    return new_xml_tree


def cut_image(src_image, new_height, new_width, origin_height, origin_width):
    horizontal_extra = int((new_width - origin_width) / 2)
    vertical_extra = int((new_height - origin_height) / 2)
    cutted_image = src_image[horizontal_extra:horizontal_extra + origin_height,
                             vertical_extra: vertical_extra + origin_width]

    return cutted_image


def resize_image(src_image, new_height, new_width, origin_height, origin_width, resize_ratio):
    resized_image = cv2.resize(src_image, (new_width, new_height))

    # resize 비율이 1보다 크면 확대이므로 가우시안 노이즈가 필요없고, 원본사이즈 유지이므로 사이즈범위 밖의 테두리 부분은 잘라냄
    if resize_ratio > 1:
        cutted_image = cut_image(resized_image, new_height, new_width, origin_height, origin_width)
        return cutted_image
    
    # resize 비율이 1보다 작으면 축소이므로 원본사이즈보다 줄어든만큼 가우시안 노이즈가 필요함
    else:
        noised_image = make_noise(resized_image, new_width, new_height, origin_height, origin_width)
        return noised_image


def walk_around_xml_files(resize_ratio, src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list):
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

                resize_tag = '_' + str(resize_ratio)
                dst_anno_file = os.path.join(dst_anno_dir_list[index], file[:-4] + resize_tag + '.xml')
                dst_img_file = os.path.join(dst_img_dir_list[index], file[:-4] + resize_tag + '.png')

                # To Do
                try:

                    print(file)
                    current_img = cv2.imread(current_img_file)
                    origin_height, origin_width, _ = current_img.shape
                    new_height, new_width = int(origin_height * resize_ratio), int(origin_width * resize_ratio)
                    resized_img = resize_image(current_img, new_height, new_width, origin_height, origin_width, resize_ratio)  # 이미지 resize
                    cv2.imwrite(dst_img_file, resized_img)
                    
                    new_xml_tree = xml_parser(root, file, new_height, new_width, origin_height, origin_width, resize_ratio)

                    # annotation 파일이 없다면 다음파일로 넘어감
                    if new_xml_tree is None:
                        continue

                    # annotation 파일이 있다면 좌표값과 이미지 함께 resize
                    else:
                        new_xml_tree.write(dst_anno_file)  # resize 좌표가 있는 xml 파일 저장

                except FileNotFoundError:
                    print('ERROR : ', file)
                    err_file_cnt += 1
                    continue
                # To DO
    print("Error : ", err_file_cnt)


if __name__ == "__main__":
    # src_dir_folder = ['train', 'test', 'coa_origin']
    src_dir_folder = ['train', 'test']
    src_dir = './sample_xml/'
    separated_dir = './output_xml/'

    # resize 를 하더라도 원본 크기는 유지하여 layer 에 입력되는 object 의 크기를 다르게 함
    # 640x640 보다 크게 확장되더라도 나머지부분은 자름
    # 640x640 보다 작게 축소되더라도 나머지부분은 가우시안 노이즈로 채움
    resize_ratio = 1.2

    src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list = make_dir_list(src_dir_folder, src_dir,
                                                                                             separated_dir)
    walk_around_xml_files(resize_ratio, src_anno_dir_list, dst_anno_dir_list, src_img_dir_list, dst_img_dir_list)

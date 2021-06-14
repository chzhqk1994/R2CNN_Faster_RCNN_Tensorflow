import numpy as np
import os
import cv2
import matplotlib as mpl
mpl.use('Agg')
import math
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree


def rotate_box_(bb, cx, cy, h, w, theta):
    new_bb = list(bb)
    for i,coord in enumerate(bb):
        # opencv calculates standard transformation matrix
        M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
        # Grab  the rotation components of the matrix)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        # Prepare the vector to be transformed
        v = [coord[0],coord[1],1]
        # Perform the actual rotation and return the image
        calculated = np.dot(M,v)
        new_bb[i] = (calculated[0],calculated[1])
    return new_bb


def rotate_box(bb, cx, cy, h, w, theta):
    new_bb = list(bb)
    rad = np.deg2rad(theta)
    for i,coord in enumerate(bb):
        # opencv calculates standard transformation matrix
        x, y = coord[0], coord[1]
        offset_x, offset_y = cx, cy
        adjusted_x = (x - offset_x)
        adjusted_y = (y - offset_y)
        cos_rad = np.cos(rad)
        sin_rad = np.sin(rad)
        xx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
        yy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
        new_bb[i] = (xx, yy)
    return new_bb

def rotate(point, origin, degrees):
    radians = np.deg2rad(degrees)
    x,y = point
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = np.cos(radians)
    sin_rad = np.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
    return qx, qy

def rotatePoint(xc, yc, xp, yp, theta):
    xoff = xp - xc
    yoff = yp - yc

    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    pResx = cosTheta * xoff + sinTheta * yoff
    pResy = - sinTheta * xoff + cosTheta * yoff
    return str(xc + pResx), str(yc + pResy)

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def build_xml_arch(theta, parsed_xml, file_path, if_debug):


    file_path = file_path.replace("annotations", "JPEGImages").replace(".xml", ".png")
    try:
        img = cv2.imread(file_path)
        (heigth_origin, width_origin) = img.shape[:2]
        (cx_origin, cy_origin) = (width_origin // 2, heigth_origin // 2)
        # img = rotate_bound(img, theta)
        # height_new, width_new = img.shape[:2]
        M = cv2.getRotationMatrix2D((cx_origin, cy_origin), theta, 1.0)
        img = cv2.warpAffine(img, M, (width_origin, heigth_origin))
        height_new, width_new = img.shape[:2]
        cv2.imwrite(os.path.join(xml_parse["output_dir_img"], xml_parse["filename"]) + "_rotate" + str(theta) +".png", img)
    except:
        return None
    print(file_path)
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
    # width.text = str(640)
    width.text = str(width_new)
    height = SubElement(size, 'height')
    # height.text = str(640)
    height.text = str(height_new)
    depth = SubElement(size, 'depth')
    depth.text = str(3)

    segmented = SubElement(annotation, 'segmented')
    segmented.text = '0'

    for obj in parsed_xml['objects']:
        # xml 태그가 robndbox 인 경우
        if not obj.find('robndbox') is None:
            label = obj.find('name').text
            cx = obj.find('./robndbox/cx').text
            cy = obj.find('./robndbox/cy').text
            width = obj.find('./robndbox/w').text
            height = obj.find('./robndbox/h').text
            angle = obj.find('./robndbox/angle').text
            try:
                x0 = obj.find('./robndbox/x0').text
                x1 = obj.find('./robndbox/x1').text
                x2 = obj.find('./robndbox/x2').text
                x3 = obj.find('./robndbox/x3').text
                y0 = obj.find('./robndbox/y0').text
                y1 = obj.find('./robndbox/y1').text
                y2 = obj.find('./robndbox/y2').text
                y3 = obj.find('./robndbox/y3').text
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
            try:
                x0_, y0_, x1_, y1_, x2_, y2_, x3_, y3_ = float(x0), float(y0), float(x1), float(y1), float(x2), float(y2), float(x3), float(y3)
                bb, cx, cy, w, h = [(x0_, y0_), (x1_, y1_), (x2_, y2_), (x3_, y3_)], float(cx), float(cy), float(width), float(height)
            except:
                bb, cx, cy, angle, width, height = [(x0_, y0_), (x1_, y1_), (x2_, y2_), (x3_, y3_)], 0, 0, 0, 0, 0

            new_bbox = rotate_box(bb, cx_origin, cy_origin, heigth_origin, width_origin, theta)
            x0, y0, x1, y1, x2, y2, x3, y3 = int(new_bbox[0][0]), int(new_bbox[0][1]), int(new_bbox[1][0]), int(new_bbox[1][1]), \
                                                     int(new_bbox[2][0]), int(new_bbox[2][1]), int(new_bbox[3][0]), int(new_bbox[3][1])

            if if_debug:
                cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), thickness=2)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                cv2.line(img, (x2, y2), (x3, y3), (0, 255, 0), thickness=2)
                cv2.line(img, (x3, y3), (x0, y0), (0, 255, 0), thickness=2)
                cv2.imwrite(os.path.join(xml_parse['sample_img_dir'], xml_parse["filename"]) + "_rotation_sample.png", img)

            x0, y0, x1, y1, x2, y2, x3, y3 = str(x0), str(y0), str(x1), str(y1), str(x2), str(y2), str(x3), str(y3)
            cx, cy = str(cx), str(cy)


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
            # robndbox 를 다루기 때문에 중심, 세로, 높이를 계산해야 하지만 bndbox 는 회전각이 0도 이기때문에 max min 값을 그대로 사용가능
            # 중심, 세로, 높이를 중요시 하는 경우 0을 계산해서 수정.
            label = obj.find('name').text
            cx = 0
            cy = 0
            width = 0
            height = 0
            angle = 0
            x0 = obj.find('./bndbox/xmin').text
            x1 = obj.find('./bndbox/xmax').text
            x2 = obj.find('./bndbox/xmax').text
            x3 = obj.find('./bndbox/xmin').text
            y0 = obj.find('./bndbox/ymin').text
            y1 = obj.find('./bndbox/ymin').text
            y2 = obj.find('./bndbox/ymax').text
            y3 = obj.find('./bndbox/ymax').text

            try:
                x0_, y0_, x1_, y1_, x2_, y2_, x3_, y3_ = float(x0), float(y0), float(x1), float(y1), float(x2), float(y2), float(x3), float(y3)
                bb, cx, cy, w, h = [(x0_, y0_), (x1_, y1_), (x2_, y2_), (x3_, y3_)], float(cx), float(cy), float(width), float(height)
            except:
                bb, cx, cy, angle, width, height = [(x0_, y0_), (x1_, y1_), (x2_, y2_), (x3_, y3_)], 0, 0, 0, 0, 0

            new_bbox = rotate_box(bb, cx_origin, cy_origin, heigth_origin, width_origin, theta)
            x0, y0, x1, y1, x2, y2, x3, y3 = int(new_bbox[0][0]), int(new_bbox[0][1]), int(new_bbox[1][0]), int(new_bbox[1][1]), \
                                                     int(new_bbox[2][0]), int(new_bbox[2][1]), int(new_bbox[3][0]), int(new_bbox[3][1])

            if if_debug:
                cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), thickness=2)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                cv2.line(img, (x2, y2), (x3, y3), (0, 255, 0), thickness=2)
                cv2.line(img, (x3, y3), (x0, y0), (0, 255, 0), thickness=2)
                cv2.imwrite(os.path.join(xml_parse['sample_img_dir'], xml_parse["filename"]) + "_rotation_sample.png", img)

            x0, y0, x1, y1, x2, y2, x3, y3 = str(x0), str(y0), str(x1), str(y1), str(x2), str(y2), str(x3), str(y3)
            cx, cy = str(cx), str(cy)


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

    result = ElementTree(annotation)
    return result


if __name__ == "__main__":

    # folder_path = "/media/qisens/2tb1/goodroof_solarpanel_parkinglot_rooftop_facility_rotation/original_test"
    folder_path = "/home/qisens/dota/R2CNN/data/io/data/ensemble/others_detail2/rebuilt"
    # train_or_test_ls = ["train", "test"]
    train_or_test_ls = ["train"]
    theta_ls = [30, 60]
    #if you want to see rotationed bbox and rotationed image, then set True (but, it takes time)
    if_debug = False


    for train_or_test in train_or_test_ls:
        for theta in theta_ls:
            anno_dir = os.path.join(folder_path, train_or_test+"/annotations")
            output_dir_images = os.path.join(folder_path, "rotation_"+str(theta)+"/"+train_or_test+"/JPEGImages")
            output_dir_annos = os.path.join(folder_path, "rotation_"+str(theta)+"/"+train_or_test+"/annotations")
            sample_dir_images = os.path.join(folder_path, "rotation_"+str(theta)+"/"+train_or_test+"/samples")

            if not os.path.exists(output_dir_images):
                os.makedirs(output_dir_images)
            if not os.path.exists(output_dir_annos):
                os.makedirs(output_dir_annos)
            if not os.path.exists(sample_dir_images):
                os.makedirs(sample_dir_images)

            for path, dirs, files in os.walk(anno_dir):
                for file in files:
                    file_path = os.path.join(path,file)
                    xml_parse = {'folder': '', 'filename': '', 'path': '', 'objects': [], 'output_dir_img':'', 'output_dir_anno':'', 'sample_img_dir':''}
                    tree = ET.parse(file_path)
                    obj_name_elements = tree.findall('./object')
                    if len(obj_name_elements) == 0:  # object 가 하나도 없다면 return None
                        print(1)
                    xml_parse['objects'] = obj_name_elements
                    bndbox_name_elements = tree.findall('*/bndbox')
                    if len(bndbox_name_elements) == 0:
                        pass
                    robndbox_name_elements = tree.findall('*/robndbox')
                    if len(robndbox_name_elements) == 0:
                        pass

                    xml_parse['folder'] = tree.getroot().find('folder').text
                    # xml_parse['filename'] = tree.getroot().find('filename').text
                    xml_parse['filename'] = os.path.splitext(file)[0]
                    xml_parse['path'] = tree.getroot().find('path').text
                    xml_parse['output_dir_img'] = output_dir_images
                    xml_parse['output_dir_anno'] = output_dir_annos
                    xml_parse['sample_img_dir'] = sample_dir_images

                    result = build_xml_arch(theta, xml_parse, file_path, if_debug)
                    anno_dest_file = os.path.join(xml_parse["output_dir_anno"], xml_parse["filename"]) + "_rotate" + str(
                        theta) + ".xml"
                    if result == None:
                        continue
                    else:
                        result.write(anno_dest_file)









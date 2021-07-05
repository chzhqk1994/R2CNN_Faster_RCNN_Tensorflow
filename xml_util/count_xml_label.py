import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, dump, ElementTree

label_list = {'goodroof': 0, 'solarpanel': 0, 'parkinglot': 0, 'facility': 0, 'rooftop': 0, 'solarpanel_slope': 0, 'solarpanel_flat': 0}

xml_dirs = '/home/qisens/Desktop/flat_slope_half/train/annotations/'

for root, dirs, files in os.walk(xml_dirs):
    for file in files:
        xml_file = os.path.join(xml_dirs, file)
        print(xml_file)
        doc = ET.parse(xml_file)
        #root 노드 가져오기
        root = doc.getroot()

        obj_tag = root.findall("object")
        print(obj_tag[0].find("name").text)

        for object in root.iter("object"):
            # print(object.find("bndbox").findtext("xmin"))
            label = object.findtext('name')
            if label in label_list:
                label_list[label] += 1

print(label_list)




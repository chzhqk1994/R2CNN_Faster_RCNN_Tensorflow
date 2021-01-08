
#!/bin/sh
export DATASET_ROOT='/media/qisens/4tb3/kowa_global/r2cnn_dataset/flatroof_solarpanel_parkinglot/'
export DATASET_NAME='ROOF'
export IMAGES='train/JPEGImages/'
export LABELS='train/annotations/'
export TEST_IMAGES='test/JPEGImages/'
export TEST_LABELS='test/annotations/'
cd ./io

python convert_data_to_tfrecord.py --VOC_dir=$DATASET_ROOT --xml_dir=$TEST_LABELS --image_dir=$TEST_IMAGES --save_name='test' --img_format='.png' --dataset=$DATASET_NAME

python convert_data_to_tfrecord.py --VOC_dir=$DATASET_ROOT --xml_dir=$LABELS --image_dir=$IMAGES --save_name='train' --img_format='.png' --dataset=$DATASET_NAME

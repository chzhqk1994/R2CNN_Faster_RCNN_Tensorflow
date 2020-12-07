
#!/bin/sh

export IMAGES=/home/qisens/works/R2CNN_Faster_RCNN_Tensorflow/tools/coa_images_with_annotation/JPEGImages/

export LABELS=/home/qisens/works/R2CNN_Faster_RCNN_Tensorflow/tools/coa_images_with_annotation/annotations/

cd /home/qisens/works/R2CNN_Faster_RCNN_Tensorflow/tools

python eval.py --img_dir=$IMAGES --image_ext='.png' --test_annotation_path=$LABELS --gpu='-1'

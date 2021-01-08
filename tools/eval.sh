#!/bin/sh

export IMAGES=$(pwd)/coa_origin/JPEGImages/

export LABELS=$(pwd)/coa_origin/annotations/

python eval.py --img_dir=$IMAGES --image_ext='.png' --test_annotation_path=$LABELS --gpu='-1'

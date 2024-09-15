import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os
from PIL import Image

# COCO 데이터셋 로드 함수
def load_coco_dataset(annotation_file, image_dir):
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    images = []
    annotations = []
    
    for img in coco_data['images']:
        image_path = os.path.join(image_dir, img['file_name'])
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        images.append(image)
        
    for ann in coco_data['annotations']:
        annotations.append(ann)
    
    return np.array(images), annotations

# 데이터 로드
train_images, train_annotations = load_coco_dataset('path/to/train_annotations.json', 'path/to/train_images')
val_images, val_annotations = load_coco_dataset('path/to/val_annotations.json', 'path/to/val_images')
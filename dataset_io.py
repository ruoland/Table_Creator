from dataset_draw import *
from dataset_constant import *
from dataset_utils import *
import os
import ujson as json


class ImageIdManager:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.id_file = os.path.join(output_dir, 'last_image_id.json')

    def get_last_image_id(self):
        last_id = 0
        for subset in ['train', 'val']:
            image_dir = os.path.join(self.output_dir, subset, 'images')
            if os.path.exists(image_dir):
                image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
                if image_files:
                    ids = [int(f.split('.')[0]) for f in image_files]
                    if ids:
                        last_id = max(last_id, max(ids))
        return last_id

    def save_last_image_id(self, last_id):
        with open(self.id_file, 'w') as f:
            json.dump({'last_image_id': last_id}, f)

    def load_last_image_id(self):
        try:
            with open(self.id_file, 'r') as f:
                data = json.load(f)
                return data['last_image_id']
        except FileNotFoundError:
            return 0
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def save_subset_results(output_dir, subset, dataset_info, coco_annotations):
    # COCO 형식 주석 저장
    images = [{"id": int(info['image_id']), 
               "file_name": f"{int(info['image_id']):06d}.png", 
               "width": int(info['image_width']), 
               "height": int(info['image_height'])} 
              for info in dataset_info]
    
    # 모든 부동소수점 값을 Python float로 변환
    def convert_floats(item):
        if isinstance(item, dict):
            return {k: convert_floats(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [convert_floats(v) for v in item]
        elif isinstance(item, (np.integer, np.floating)):
            return float(item)
        return item

    coco_annotations = convert_floats(coco_annotations)
    
    coco_format = {
        "images": images,
        "annotations": coco_annotations,
        "categories": [
            {"id": 0, "name": "cell"},
            {"id": 1, "name": "table"}
        ]
    }
    
    annotation_file = os.path.join(output_dir, f'{subset}_annotations.json')
    
    with open(annotation_file, 'w') as f:
        json.dump(coco_format, f, cls=NumpyEncoder)
    logger.info(f"Saved annotations for {subset} to {annotation_file}")

def validate_cell(cell):
    if len(cell) < 7:
        return False
    x1, y1, x2, y2 = cell[:4]
    return (x2 > x1 and y2 > y1 and 
            x2 - x1 >= 1 and y2 - y1 >= 1 and
            all(isinstance(coord, int) for coord in [x1, y1, x2, y2]))

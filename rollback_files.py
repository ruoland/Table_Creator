from opti_calc import *
from rollback_draw import *
from rollback_constant import *
from rollback_utils import *
import os, csv
from datetime import datetime
import yaml
import itertools

import random, json



def determine_subset(train_ratio, val_ratio):
    rand = random.random()
    if rand < train_ratio:
        return 'train'
    elif rand < train_ratio + val_ratio:
        return 'val'
    else:
        return 'test'

def save_dataset_info(output_dir, dataset_info, stats):
    total_images = sum(len(info) for info in dataset_info.values())
    print(f"총 생성된 이미지 수: {total_images}")

    for subset, info in dataset_info.items():
        print(f"{subset.capitalize()} 세트:")
        print(f"  총 이미지 수: {len(info)}")
        print(f"  밝은 이미지 수: {sum(1 for item in info if item['bg_mode'] == 'light')}")
        print(f"  어두운 이미지 수: {sum(1 for item in info if item['bg_mode'] == 'dark')}")
        print(f"  간격 있는 이미지 수: {sum(1 for item in info if item['has_gap'])}")
        print(f"  불완전 이미지 수: {sum(1 for item in info if item['is_imperfect'])}")
        print()

    yaml_content = {
        'train': os.path.join('train', 'images'),
        'val': os.path.join('val', 'images'),
        'nc': 5,
        'names': ['cell', 'merged_cell', 'row', 'column', 'table']
    }
    
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_content, f)

    with open(os.path.join(output_dir, 'dataset_stats.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=stats[0].keys())
        writer.writeheader()
        writer.writerows(stats)

    summary_stats = calculate_summary_stats(stats)
    with open(os.path.join(output_dir, 'summary_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=4, ensure_ascii=False)
        
def save_coco_annotations(output_dir, dataset_info, coco_annotations, subset):
    coco_data = {
        "info": {
            "description": f"Table Detection Dataset - {subset}",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        },
        "licenses": [
            {
                "url": "",
                "id": 1,
                "name": "Unknown"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "cell", "supercategory": "table"},
            {"id": 2, "name": "merged_cell", "supercategory": "table"},
            {"id": 3, "name": "row", "supercategory": "table"},
            {"id": 4, "name": "column", "supercategory": "table"},
            {"id": 5, "name": "table", "supercategory": "table"}
        ]
    }

    image_ids = set()
    for image_info in dataset_info[subset]:
        # 이미지 경로를 'train/images/' 또는 'val/images/'로 설정
        coco_data["images"].append({
            "id": image_info['image_id'],
            "width": image_info['image_width'],
            "height": image_info['image_height'],
            "file_name": os.path.join(subset, "images", f"image_{image_info['image_id']:06d}.png"),  # 수정된 경로
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        })
        image_ids.add(image_info['image_id'])

    # 계층 구조를 위한 임시 저장소
    tables = {}
    rows = {}
    columns = {}

    for ann in coco_annotations:
        if ann["image_id"] in image_ids:
            if ann["category_id"] == 5:  # Table
                tables[ann["id"]] = ann
                ann["rows"] = []
                ann["columns"] = []
            elif ann["category_id"] == 3:  # Row
                rows[ann["id"]] = ann
                ann["cells"] = []
            elif ann["category_id"] == 4:  # Column
                columns[ann["id"]] = ann
                ann["cells"] = []
            elif ann["category_id"] in [1, 2]:  # Cell or Merged Cell
                if "row_id" in ann:
                    if ann["row_id"] not in rows:
                        logger.warning(f"Row ID {ann['row_id']} not found for cell {ann['id']}")
                        rows[ann["row_id"]] = {"cells": []}
                    rows[ann["row_id"]]["cells"].append(ann["id"])
                if "column_id" in ann:
                    if ann["column_id"] not in columns:
                        logger.warning(f"Column ID {ann['column_id']} not found for cell {ann['id']}")
                        columns[ann["column_id"]] = {"cells": []}
                    columns[ann["column_id"]]["cells"].append(ann["id"])
        # 두 번째 패스: 테이블에 행과 열 연결
        for table in tables.values():
            table["rows"] = []
            table["columns"] = []

        for row in rows.values():
            assigned = False
            for table in tables.values():
                if (row["bbox"][0] >= table["bbox"][0] and 
                    row["bbox"][1] >= table["bbox"][1] and
                    row["bbox"][0] + row["bbox"][2] <= table["bbox"][0] + table["bbox"][2] and
                    row["bbox"][1] + row["bbox"][3] <= table["bbox"][1] + table["bbox"][3]):
                    table["rows"].append(row["id"])
                    assigned = True
                    break

        for column in columns.values():
            assigned = False
            for table in tables.values():
                if (column["bbox"][0] >= table["bbox"][0] and 
                    column["bbox"][1] >= table["bbox"][1] and
                    column["bbox"][0] + column["bbox"][2] <= table["bbox"][0] + table["bbox"][2] and
                    column["bbox"][1] + column["bbox"][3] <= table["bbox"][1] + table["bbox"][3]):
                    table["columns"].append(column["id"])
                    assigned = True
                    break
            if not assigned:
                logger.warning(f"Column {column['id']} could not be assigned to any table")

        # 테이블, 행, 열의 일관성 검사
        for table in tables.values():
            if not table["rows"]:
                logger.warning(f"Table {table['id']} has no rows")
            if not table["columns"]:
                logger.warning(f"Table {table['id']} has no columns")

        # 최종 어노테이션 리스트 생성
        coco_data["annotations"] = list(tables.values()) + list(rows.values()) + list(columns.values()) + [
            ann for ann in coco_annotations 
            if ann["image_id"] in image_ids and ann["category_id"] in [1, 2]
        ]

        # 어노테이션 검증 및 조정
#        validate_and_adjust_annotations(coco_data["annotations"], coco_data["images"])

        # COCO 데이터 저장
        with open(os.path.join(output_dir, f'{subset}_annotations.json'), 'w') as f:
            json.dump(coco_data, f, indent=4)  # 가독성을 위해 indent 추가

        logger.info(f"Saved COCO annotations for {subset} subset")

def adjust_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))
    w = min(w, img_width - x)
    h = min(h, img_height - y)
    return [x, y, w, h]
def validate_and_adjust_annotations(annotations, images):
    for ann in annotations:
        img_info = next(img for img in images if img['id'] == ann['image_id'])
        ann['bbox'] = adjust_bbox(ann['bbox'], img_info['width'], img_info['height'])
def save_image_and_annotations(img, image_stats, output_dir, subset):
    img_filename = f"image_{image_stats['image_id']:06d}.png"
    img_path = os.path.join(output_dir, subset, 'images', img_filename)
    img.save(img_path)
import gc

def save_last_image_id(output_dir, last_ids):
    with open(os.path.join(output_dir, 'last_image_ids.json'), 'w') as f:
        json.dump(last_ids, f)

def load_last_image_ids(output_dir):
    try:
        with open(os.path.join(output_dir, 'last_image_ids.json'), 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'train': 0, 'val': 0}
def get_last_image_id(output_dir, subset):
    last_id = 0
    image_dir = os.path.join(output_dir, subset, 'images')
    if os.path.exists(image_dir):
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        if image_files:
            last_file = max(image_files)
            last_id = int(last_file[6:-4])  # 'table_00123.png' -> 123
    return last_id
def get_last_image_id(output_dir):
    last_id = 0
    for subset in ['train', 'val', 'test']:
        image_dir = os.path.join(output_dir, subset, 'images')
        if os.path.exists(image_dir):
            image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
            if image_files:
                last_file = max(image_files)
                last_id = max(last_id, int(last_file[6:-4]))  # 'table_00123.png' -> 123
    return last_id

def save_last_image_id(output_dir, last_id):
    with open(os.path.join(output_dir, 'last_image_id.json'), 'w') as f:
        json.dump({'last_image_id': last_id}, f)

def load_last_image_id(output_dir):
    try:
        with open(os.path.join(output_dir, 'last_image_id.json'), 'r') as f:
            data = json.load(f)
            return data['last_image_id']
    except FileNotFoundError:
        return 0
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
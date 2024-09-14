import os
import random
import json
import gc
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import logging
from grrr_common import generate_image_and_labels, generate_random_resolution, determine_subset
from grrr_utils import create_directory
from grrr_constants import DAYS, TIMES, GRADES

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_header_info(col):
    if col == 0:  # 첫 번째 열
        header_type = random.choice(['day', 'time'])
        if header_type == 'day':
            return random.choice(DAYS), 'day'
        else:
            return random.choice(TIMES), 'time'
    else:  # 나머지 열
        return random.choice(GRADES), 'grade'
def generate_coco_annotations(cells, table_bbox, image_width, image_height, rows, cols, image_id, annotation_id):
    annotations = []
    
    # Cell annotations
    for cell in cells:
        bbox = [cell[0], cell[1], cell[2] - cell[0], cell[3] - cell[1]]
        
        if len(cell) > 7:  # 헤더 셀
            category_id = 8
            attributes = {
                'row_start': cell[4],
                'column_start': cell[5],
                'text': cell[6],
                'header_type': cell[7]
            }
        elif len(cell) > 6:  # 병합된 셀
            category_id = 2
            attributes = {
                'row_start': cell[4],
                'column_start': cell[5],
                'row_end': cell[6],
                'column_end': cell[7]
            }
        else:  # 일반 셀
            category_id = 1
            attributes = {
                'row_start': cell[4],
                'column_start': cell[5]
            }
        
        cell_annotation = {
            'id': annotation_id,
            'image_id': image_id,
            'category_id': category_id,
            'bbox': bbox,
            'area': bbox[2] * bbox[3],
            'iscrowd': 0,
            'attributes': attributes
        }
        
        annotations.append(cell_annotation)
        annotation_id += 1
    
    # Row and Column annotations
    for entity_type, category_id in [('row', 3), ('column', 4)]:
        for i in range(rows if entity_type == 'row' else cols):
            entity_cells = [cell for cell in cells if cell[4 if entity_type == 'row' else 5] == i]
            if entity_cells:
                min_coord = min(cell[0 if entity_type == 'row' else 1] for cell in entity_cells)
                max_coord = max(cell[2 if entity_type == 'row' else 3] for cell in entity_cells)
                fixed_coord = entity_cells[0][1 if entity_type == 'row' else 0]
                size = entity_cells[0][3 if entity_type == 'row' else 2] - fixed_coord
                bbox = [min_coord, fixed_coord, max_coord - min_coord, size] if entity_type == 'row' else [fixed_coord, min_coord, size, max_coord - min_coord]
                
                annotations.append({
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': category_id,
                    'bbox': bbox,
                    'area': bbox[2] * bbox[3],
                    'iscrowd': 0
                })
                annotation_id += 1
    
    # Row and Column count annotations
    for count_type, category_id, count in [('row', 5, rows), ('column', 6, cols)]:
        annotations.append({
            'id': annotation_id,
            'image_id': image_id,
            'category_id': category_id,
            'bbox': [0, 0, image_width, image_height],
            'area': image_width * image_height,
            'iscrowd': 0,
            'attributes': {'count': count}
        })
        annotation_id += 1
    
    # Table annotation
    annotations.append({
        'id': annotation_id,
        'image_id': image_id,
        'category_id': 7,
        'bbox': table_bbox,
        'area': table_bbox[2] * table_bbox[3],
        'iscrowd': 0,
        'attributes': {'rows': rows, 'columns': cols}
    })
    annotation_id += 1
    
    return annotations, annotation_id
def process_image(args):
    image_id, resolution, bg_mode, has_gap, is_imperfect, output_dir = args
    img, cells, table_bbox, image_stats = generate_image_and_labels(
        image_id, resolution, bg_mode, has_gap, is_imperfect
    )
    
    if img is None or cells is None or table_bbox is None or image_stats is None:
        logger.warning(f"Failed to generate image {image_id}")
        return None
    
    img_filename = f"table_{image_id:06d}.png"
    img_path = os.path.join(output_dir, 'images', img_filename)
    img.save(img_path)
    
    return cells, table_bbox, image_stats, img_filename
def generate_large_dataset_coco(output_dir, num_images, imperfect_ratio=0.3, train_ratio=0.8, val_ratio=0.1):
    create_directory(output_dir)
    create_directory(os.path.join(output_dir, 'images'))
    
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "cell"},
            {"id": 2, "name": "merged_cell"},
            {"id": 3, "name": "row"},
            {"id": 4, "name": "column"},
            {"id": 5, "name": "row_count"},
            {"id": 6, "name": "column_count"},
            {"id": 7, "name": "table"},
            {"id": 8, "name": "header_cell"}
        ]
    }
    
    image_id = 1
    annotation_id = 1
    
    with ProcessPoolExecutor() as executor:
        args_list = [
            (i, generate_random_resolution(), random.choice(['light', 'dark']),
             random.choice([True, False]), random.random() < imperfect_ratio,
             output_dir)
            for i in range(1, num_images + 1)
        ]
        
        results = list(tqdm(executor.map(process_image, args_list), total=num_images, desc="Generating COCO dataset"))
        
    for result in results:
        if result is None:
            continue
        
        cells, table_bbox, image_stats, img_filename = result
        
        coco_output["images"].append({
            "id": image_id,
            "file_name": img_filename,
            "width": image_stats['image_width'],
            "height": image_stats['image_height']
        })
        
        annotations, annotation_id = generate_coco_annotations(
            cells, table_bbox, image_stats['image_width'], image_stats['image_height'], 
            image_stats['rows'], image_stats['cols'], image_id, annotation_id
        )
        coco_output["annotations"].extend(annotations)
        
        image_id += 1
        
        if image_id % 1000 == 0:
            gc.collect()
    
    # 데이터셋 분할
    num_train = int(num_images * train_ratio)
    num_val = int(num_images * val_ratio)
    
    random.shuffle(coco_output["images"])
    
    train_images = coco_output["images"][:num_train]
    val_images = coco_output["images"][num_train:num_train+num_val]
    test_images = coco_output["images"][num_train+num_val:]
    
    for subset, images in [("train", train_images), ("val", val_images), ("test", test_images)]:
        subset_output = {
            "images": images,
            "annotations": [ann for ann in coco_output["annotations"] if ann["image_id"] in [img["id"] for img in images]],
            "categories": coco_output["categories"]
        }
        
        with open(os.path.join(output_dir, f'{subset}_annotations.json'), 'w') as f:
            json.dump(subset_output, f)
        
        # 이미지 파일 이동
        create_directory(os.path.join(output_dir, subset, 'images'))
        for img in images:
            os.rename(os.path.join(output_dir, 'images', img["file_name"]),
                      os.path.join(output_dir, subset, 'images', img["file_name"]))
    
    logger.info("COCO annotations saved successfully.")

if __name__ == "__main__":
    output_dir = 'table_dataset_coco'
    num_images = 1000
    imperfect_ratio = 0.1
    train_ratio = 0.8
    val_ratio = 0.1
    generate_large_dataset_coco(output_dir, num_images, imperfect_ratio, train_ratio, val_ratio)
    logger.info("COCO dataset generation completed.")
from typing import Tuple, List, Any
from PIL import Image
import os
import random, sys
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw
from dataset_io import save_subset_results, compress_and_save_image
from dataset_config import  TableGenerationConfig
from dataset_table import generate_image_and_labels 
from multiprocessing import Pool, cpu_count
from logging_config import  table_logger

def batch_dataset(output_dir, total_num_images, train_ratio=0.8, num_processes=None, config=None):
    if num_processes is None:
        num_processes = cpu_count()
    
    table_logger.info(f"Using {num_processes} processes")
    
    os.makedirs(output_dir, exist_ok=True)
    for subset in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, subset, 'images'), exist_ok=True)

    tasks = []
    train_images = int(total_num_images * train_ratio)

    def create_tasks(start, end, subset):
        image_ids = list(range(start, end))
        random.shuffle(image_ids)  # 이미지 ID를 섞습니다
        chunk_size = max(1, len(image_ids) // num_processes)
        for i in range(0, len(image_ids), chunk_size):
            chunk = image_ids[i:i+chunk_size]
            tasks.append((chunk, output_dir, subset, config))
            table_logger.debug(f"Created task: {chunk} for {subset}")

    create_tasks(0, train_images, 'train')
    create_tasks(train_images, total_num_images, 'val')

    table_logger.info(f"Created {len(tasks)} tasks")

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(process_images, tasks), total=len(tasks), desc="Processing images"))

    all_dataset_info = {'train': [], 'val': []}
    all_coco_annotations = {'train': [], 'val': []}

    for dataset_info, coco_annotations in results:
        subset = dataset_info[0]['subset'] if dataset_info else 'train'
        all_dataset_info[subset].extend(dataset_info)
        all_coco_annotations[subset].extend(coco_annotations)

    for subset in ['train', 'val']:
        save_subset_results(output_dir, subset, all_dataset_info[subset], all_coco_annotations[subset])

    table_logger.info("Dataset generation completed.")
    for subset in ['train', 'val']:
        files = len([f for f in os.listdir(os.path.join(output_dir, subset, 'images')) if f.endswith('.jpeg')])
        table_logger.info(f"{subset.capitalize()} files generated: {files}")

    return all_dataset_info, all_coco_annotations

def process_images(args: Tuple[List[int], str, str, float, TableGenerationConfig]):
    import os

    image_ids, output_dir, subset, config = args
    table_logger.info(f"Processing images {min(image_ids)} to {max(image_ids)} for {subset}")

    dataset_info = []
    coco_annotations = []

    for image_id in image_ids:
        if config.config_mode == 'Random':
            config.randomize_settings()
        elif config.config_mode == 'None':
            config.disable_all_effects()
        table_logger.debug(f"Generating image {image_id}")
        img, annotations, actual_width, actual_height = generate_image_and_labels(
            image_id,
            random.choice(['light', 'dark']),
            random.choice([True, False]),
            random.random() < config.imperfect_ratio,
            config
        )
        
        if img is not None:
            img_path = os.path.join(output_dir, subset, 'images', f"{image_id:06d}.jpeg")
            compress_and_save_image(img, img_path, quality=80)
            
            # 이미지 크기와 어노테이션 비교
            for ann in annotations:
                bbox = ann['bbox']
                if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] > actual_width or bbox[1] + bbox[3] > actual_height:
                    table_logger.warning(f"Annotation out of bounds for image {image_id}: {bbox}")

            image_info = {
                'image_id': image_id,
                'image_width': actual_width,
                'image_height': actual_height,
                'subset': subset,
                'file_name': f"{image_id:06d}.jpeg"
            }
            dataset_info.append(image_info)
            coco_annotations.extend(annotations)
            table_logger.debug(f"Image saved: {img_path}")
            
            # 메모리 해제
            del img
        else:
            table_logger.warning(f"Failed to generate image: ID {image_id} for {subset}")

    table_logger.info(f"Completed processing images for {subset}")
    
    return dataset_info, coco_annotations
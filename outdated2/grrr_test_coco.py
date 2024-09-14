import os
import json
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_annotations(dataset_dir, mode='random', num_samples=5, specific_files=None):
    # COCO 주석 파일 로드
    with open(os.path.join(dataset_dir, 'annotations.json'), 'r') as f:
        coco_data = json.load(f)

    # 이미지 ID와 파일 이름 매핑
    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    filename_to_image = {img['file_name']: img for img in coco_data['images']}

    # 카테고리 ID와 이름 매핑
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    if mode == 'random':
        selected_images = random.sample(coco_data['images'], num_samples)
    elif mode == 'specific':
        selected_images = [filename_to_image[f] for f in specific_files if f in filename_to_image]
    else:
        raise ValueError("Invalid mode. Choose 'random' or 'specific'.")

    for img_info in selected_images:
        img_id = img_info['id']
        img_filename = img_info['file_name']
        
        # 이미지 로드
        img_path = os.path.join(dataset_dir, 'train', img_filename)  # 'train' 폴더에서 찾습니다. 필요에 따라 'val' 또는 'test'로 변경하세요.
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        
        img = Image.open(img_path)
        
        # 그림 설정
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img)
        
        # 해당 이미지의 주석 찾기
        # 해당 이미지의 주석 찾기
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
        
        column_count = None
        
        # 바운딩 박스 그리기
        for ann in annotations:
            category = category_id_to_name[ann['category_id']]
            bbox = ann['bbox']
            
            if category == 'column_count':
                column_count = ann['attributes'].get('count', None)
                continue  # column_count는 별도로 처리하므로 여기서 건너뜁니다.
            
            if category == 'merged_cell':
                color = 'blue'
                row_start = ann['attributes']['row_start']
                col_start = ann['attributes']['column_start']
                row_end = ann['attributes']['row_end']
                col_end = ann['attributes']['column_end']
                label = f"Merged: ({row_start},{col_start})-({row_end},{col_end})"
            elif category == 'cell':
                color = 'red'
                label = 'Cell'
            elif category == 'header_cell':
                color = 'green'
                label = 'Header'
            else:
                color = 'yellow'
                label = category
            
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                     linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # 라벨 표시
            ax.text(bbox[0], bbox[1]-5, label, color=color, fontsize=8, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # column_count 정보 표시
        if column_count is not None:
            ax.text(10, 10, f"Column Count: {column_count}", color='purple', fontsize=12, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        ax.set_title(f"Image ID: {img_id}, Filename: {img_filename}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# 사용 예:
dataset_dir = 'table_dataset_coco'

# 랜덤 선택
visualize_annotations(dataset_dir, mode='random', num_samples=3)

# 특정 파일 선택
specific_files = ['table_000001.png', 'table_000008.png', 'table_000032.png']
visualize_annotations(dataset_dir, mode='specific', specific_files=specific_files)

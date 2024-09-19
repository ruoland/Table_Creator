

import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

def validate_annotations(image_path, annotation_path, output_dir, specific_image_id=None):
    # COCO 주석 파일 로드
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    # 이미지 ID와 파일 이름 매핑
    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}

    # 카테고리 ID와 이름 매핑
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # 특정 이미지 ID가 지정된 경우
    if specific_image_id is not None:
        selected_images = [img for img in coco_data['images'] if img['id'] == specific_image_id]
        if not selected_images:
            print(f"Image with ID {specific_image_id} not found.")
            return
    else:
        # 무작위로 10개의 이미지 선택 (또는 전체 이미지 수가 10개 미만이면 전체 선택)
        selected_images = random.sample(coco_data['images'], min(10, len(coco_data['images'])))

    for image_info in selected_images:
        image_id = image_info['id']
        filename = image_id_to_filename[image_id]
        img_path = os.path.join(image_path, filename)

        # 이미지 로드
        img = Image.open(img_path)

        # 해당 이미지의 주석 찾기
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

        # 각 카테고리별로 별도의 이미지 생성
        for category_id, category_name in category_id_to_name.items():
            fig, ax = plt.subplots(1)
            ax.imshow(img)
            ax.set_title(f"Image ID: {image_id}, Category: {category_name}")

            # 현재 카테고리에 해당하는 주석만 필터링
            category_annotations = [ann for ann in annotations if ann['category_id'] == category_id]

            for ann in category_annotations:
                bbox = ann['bbox']
                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                         linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

            # 카테고리별 이미지 저장
            output_path = os.path.join(output_dir, f"validated_{image_id}_{category_name}.png")
            plt.savefig(output_path)
            plt.close(fig)

        print(f"Validated image saved for Image ID: {image_id}")

    print("Validation complete. Please check the output images.")

def check_annotation_consistency(annotation_path):
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    image_ids = set(img['id'] for img in coco_data['images'])
    annotation_image_ids = set(ann['image_id'] for ann in coco_data['annotations'])
    category_ids = set(cat['id'] for cat in coco_data['categories'])
    annotation_category_ids = set(ann['category_id'] for ann in coco_data['annotations'])

    print("Checking annotation consistency...")
    print(f"Number of images: {len(coco_data['images'])}")
    print(f"Number of annotations: {len(coco_data['annotations'])}")
    print(f"Number of categories: {len(coco_data['categories'])}")
    print(f"All annotation image IDs exist in images: {annotation_image_ids.issubset(image_ids)}")
    print(f"All annotation category IDs exist in categories: {annotation_category_ids.issubset(category_ids)}")

    # 추가적인 검증
    for ann in coco_data['annotations']:
        img_info = next(img for img in coco_data['images'] if img['id'] == ann['image_id'])
        bbox = ann['bbox']
        if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] > img_info['width'] or bbox[1] + bbox[3] > img_info['height']:
            print(f"Warning: Bounding box out of image bounds for annotation {ann['id']}")

    print("Consistency check complete.")

if __name__ == "__main__":
    image_path = r"yolox_table_dataset_simple233-0919-2\train\images"
    annotation_path = r"yolox_table_dataset_simple233-0919-2\train_annotations.json"
    output_dir = r"valid"

    os.makedirs(output_dir, exist_ok=True)

    specific_image_id = int(input("Enter the specific image ID to validate (or press Enter to validate random images): ") or 0)
    
    validate_annotations(image_path, annotation_path, output_dir, specific_image_id if specific_image_id != 0 else None)
    check_annotation_consistency(annotation_path)

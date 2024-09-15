import json
import os
from PIL import Image
import argparse
from tqdm import tqdm

def validate_dataset(dataset_dir, ann_file):
    with open(ann_file, 'r') as f:
        ann_data = json.load(f)

    print(f"Validating dataset: {dataset_dir}")
    print(f"Using annotation file: {ann_file}")

    error_log = []

    # 1. 어노테이션 파일 구조 확인
    required_keys = ['images', 'annotations', 'categories']
    for key in required_keys:
        if key not in ann_data:
            error_msg = f"Error: '{key}' is missing in the annotation file."
            print(error_msg)
            error_log.append(error_msg)
            return

    # 이미지 ID와 파일 이름 매핑
    image_id_to_file = {img['id']: img['file_name'] for img in ann_data['images']}

    # 2 & 3. 이미지 파일 존재 및 경로 일치 여부 확인
    missing_images = []
    for img in tqdm(ann_data['images'], desc="Checking image files"):
        img_path = os.path.join(dataset_dir, img['file_name'])
        if not os.path.exists(img_path):
            missing_images.append(img['file_name'])

    if missing_images:
        error_msg = f"Error: {len(missing_images)} image(s) are missing:"
        print(error_msg)
        error_log.append(error_msg)
        for img in missing_images[:10]:
            error_log.append(f"  {img}")
        if len(missing_images) > 10:
            error_log.append(f"  ... and {len(missing_images) - 10} more.")
    else:
        print("All image files exist and match the annotations.")

    # 4. 누락된 어노테이션 확인
    image_ids_with_annotations = set(ann['image_id'] for ann in ann_data['annotations'])
    images_without_annotations = set(img['id'] for img in ann_data['images']) - image_ids_with_annotations

    if images_without_annotations:
        warning_msg = f"Warning: {len(images_without_annotations)} image(s) have no annotations:"
        print(warning_msg)
        error_log.append(warning_msg)
        for img_id in list(images_without_annotations)[:10]:
            error_log.append(f"  {image_id_to_file[img_id]}")
        if len(images_without_annotations) > 10:
            error_log.append(f"  ... and {len(images_without_annotations) - 10} more.")

    # 5 & 6. 이미지 크기와 바운딩 박스 좌표 유효성 검사
    invalid_annotations = []
    for ann in tqdm(ann_data['annotations'], desc="Validating annotations"):
        img_info = next(img for img in ann_data['images'] if img['id'] == ann['image_id'])
        img_path = os.path.join(dataset_dir, img_info['file_name'])
        
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            error_msg = f"Error opening image {img_path}: {str(e)}"
            print(error_msg)
            error_log.append(error_msg)
            continue

        if width != img_info['width'] or height != img_info['height']:
            warning_msg = f"Warning: Image size mismatch for {img_info['file_name']}"
            print(warning_msg)
            error_log.append(warning_msg)
            error_log.append(f"  Annotation: {img_info['width']}x{img_info['height']}, Actual: {width}x{height}")

        bbox = ann['bbox']
        if (bbox[0] < 0 or bbox[1] < 0 or 
            bbox[0] + bbox[2] > width or 
            bbox[1] + bbox[3] > height):
            invalid_annotations.append((ann['id'], img_info['file_name']))

    if invalid_annotations:
        error_msg = f"Error: {len(invalid_annotations)} invalid bounding box(es) found:"
        print(error_msg)
        error_log.append(error_msg)
        for ann_id, img_file in invalid_annotations[:10]:
            error_log.append(f"  Annotation ID: {ann_id}, Image: {img_file}")
        if len(invalid_annotations) > 10:
            error_log.append(f"  ... and {len(invalid_annotations) - 10} more.")
    else:
        print("All bounding boxes are valid.")

    print("Dataset validation completed.")

    # 오류 로그 저장
    if error_log:
        error_log_file = os.path.join(os.path.dirname(ann_file), "validation_errors.log")
        with open(error_log_file, 'w') as f:
            f.write('\n'.join(error_log))
        print(f"Error log saved to: {error_log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate COCO format dataset")
    parser.add_argument("dataset_dir", type=str, help="Path to the dataset directory")
    parser.add_argument("ann_file", type=str, help="Path to the annotation file")
    args = parser.parse_args()

    validate_dataset(args.dataset_dir, args.ann_file)

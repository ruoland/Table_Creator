import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
def validate_annotations(image_path, annotation_path, output_dir, specific_image_id=None):
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    print("\n=== JSON 데이터 샘플 ===")
    print("이미지 데이터 샘플:")
    print(json.dumps(coco_data['images'][:2], indent=2, ensure_ascii=False))
    print("\n카테고리 데이터 샘플:")
    print(json.dumps(coco_data['categories'][:2], indent=2, ensure_ascii=False))
    print("\n어노테이션 데이터 샘플:")
    print(json.dumps(coco_data['annotations'][:2], indent=2, ensure_ascii=False))

    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    if specific_image_id is not None:
        selected_images = [img for img in coco_data['images'] if img['id'] == specific_image_id]
        if not selected_images:
            print(f"이미지 ID {specific_image_id}를 찾을 수 없습니다.")
            return
    else:
        selected_images = random.sample(coco_data['images'], min(10, len(coco_data['images'])))

    for image_info in selected_images:
        image_id = image_info['id']
        filename = image_id_to_filename[image_id]
        img_path = os.path.join(image_path, filename)

        print(f"\n=== 선택된 이미지 정보 (ID: {image_id}) ===")
        print("이미지 메타데이터:")
        print(json.dumps(image_info, indent=2, ensure_ascii=False))

        img = Image.open(img_path)
        print(f"이미지 크기: {img.size}")
        print(f"이미지 모드: {img.mode}")

        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
        print(f"\n이 이미지의 어노테이션 수: {len(annotations)}")
        print("어노테이션 데이터:")
        print(json.dumps(annotations, indent=2, ensure_ascii=False))

        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img)
        ax.set_title(f"Image ID: {image_id}")

        for ann in annotations:
            bbox = ann['bbox']
            category_name = category_id_to_name[ann['category_id']]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(bbox[0], bbox[1], category_name, color='r', fontsize=8, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        output_path = os.path.join(output_dir, f"validated_{image_id}.png")
        plt.savefig(output_path)
        plt.close(fig)

        print(f"검증된 이미지가 저장되었습니다. 이미지 ID: {image_id}")

    print("검증이 완료되었습니다. 출력된 이미지를 확인해주세요.")

def check_annotation_consistency(annotation_path):
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    print("\n=== JSON 데이터 통계 ===")
    print(f"이미지 수: {len(coco_data['images'])}")
    print(f"어노테이션 수: {len(coco_data['annotations'])}")
    print(f"카테고리 수: {len(coco_data['categories'])}")

    print("\n이미지 ID 샘플:")
    print(json.dumps([img['id'] for img in coco_data['images'][:5]], indent=2))
    print("\n카테고리 ID 샘플:")
    print(json.dumps([cat['id'] for cat in coco_data['categories']], indent=2))

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

    for ann in coco_data['annotations']:
        img_info = next(img for img in coco_data['images'] if img['id'] == ann['image_id'])
        bbox = ann['bbox']
        if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] > img_info['width'] or bbox[1] + bbox[3] > img_info['height']:
            print(f"Warning: Bounding box out of image bounds for annotation {ann['id']}")

    print("Consistency check complete.")
def add_images_and_annotations(annotation_path):
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    num_images = len(coco_data['images'])
    num_annotations = len(coco_data['annotations'])

    print("\n=== 이미지 및 어노테이션 통계 ===")
    print(f"총 이미지 수: {num_images}")
    print(f"총 어노테이션 수: {num_annotations}")

    print("\n이미지 데이터 샘플:")
    print(json.dumps(coco_data['images'][:2], indent=2, ensure_ascii=False))
    print("\n어노테이션 데이터 샘플:")
    print(json.dumps(coco_data['annotations'][:2], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    image_path = r"C:\project\table2\train\images"
    annotation_path = r"C:\project\table2\train_annotations.json"
    output_dir = r"valid"

    os.makedirs(output_dir, exist_ok=True)

    while True:
        print("\nMenu:")
        print("1. Validate annotations")
        print("2. Check annotation consistency")
        print("3. Add images and annotations count")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            specific_image_id = int(input("Enter the specific image ID to validate (or press Enter to validate random images): ") or 0)
            validate_annotations(image_path, annotation_path, output_dir, specific_image_id if specific_image_id != 0 else None)
        elif choice == '2':
            check_annotation_consistency(annotation_path)
        elif choice == '3':
            add_images_and_annotations(annotation_path)
        elif choice == '4':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")

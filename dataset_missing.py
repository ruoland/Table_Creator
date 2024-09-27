import os
import json
from tqdm import tqdm
from PIL import Image
from dataset_creator import generate_image_and_labels, generate_random_resolution

def validate_and_fix_dataset(output_dir, target_counts):
    for subset, target_count in target_counts.items():
        print(f"Validating and fixing {subset} dataset...")
        
        image_dir = os.path.join(output_dir, subset, 'images')
        anno_path = os.path.join(output_dir, f'{subset}_annotations.json')
        info_path = os.path.join(output_dir, f'{subset}_dataset_info.json')

        # 어노테이션 파일 로드
        with open(anno_path, 'r') as f:
            anno_data = json.load(f)
        
        # 데이터셋 정보 파일 로드
        with open(info_path, 'r') as f:
            info_data = json.load(f)

        # 이미지 파일 목록
        image_files = set(f for f in os.listdir(image_dir) if f.endswith('.png'))

        # 어노테이션에 있는 이미지 목록
        annotated_images = set(img['file_name'] for img in anno_data['images'])

        print(f"Actual image files: {len(image_files)}")
        print(f"Images in annotations: {len(annotated_images)}")

        # 부족한 이미지 또는 어노테이션 채우기
        missing_annotations = image_files - annotated_images
        missing_images = annotated_images - image_files

        # 어노테이션 없는 이미지에 대한 어노테이션 생성
        if missing_annotations:
            print(f"Generating annotations for {len(missing_annotations)} images")
            for img_file in tqdm(missing_annotations, desc="Generating annotations"):
                image_id = int(img_file.split('.')[0])
                img_path = os.path.join(image_dir, img_file)
                with Image.open(img_path) as img:
                    width, height = img.size
                anno_data['images'].append({
                    'id': image_id,
                    'file_name': img_file,
                    'width': width,
                    'height': height
                })
                # 여기에 필요한 경우 객체 어노테이션 생성 로직 추가

        # 이미지 없는 어노테이션에 대한 이미지 생성
        if missing_images:
            print(f"Generating {len(missing_images)} missing images")
            for img_file in tqdm(missing_images, desc="Generating images"):
                image_id = int(img_file.split('.')[0])
                img_info = next(img for img in anno_data['images'] if img['file_name'] == img_file)
                resolution = (img_info['width'], img_info['height'])
                img, annotations, image_stats = generate_image_and_labels(
                    image_id, resolution, (0, 0, 0, 0), 'light', False, False
                )
                if img is not None:
                    img.save(os.path.join(image_dir, img_file))

        # 목표 이미지 수에 맞추기
        current_image_count = len(anno_data['images'])
        if current_image_count < target_count:
            print(f"Generating {target_count - current_image_count} additional images")
            for i in tqdm(range(current_image_count, target_count), desc="Generating additional images"):
                image_id = max(img['id'] for img in anno_data['images']) + 1
                resolution, margins = generate_random_resolution()
                img, annotations, image_stats = generate_image_and_labels(
                    image_id, resolution, margins, 'light', False, False
                )
                if img is not None:
                    img_filename = f"{image_id:06d}.png"
                    img.save(os.path.join(image_dir, img_filename))
                    anno_data['images'].append({
                        'id': image_id,
                        'file_name': img_filename,
                        'width': resolution[0],
                        'height': resolution[1]
                    })
                    anno_data['annotations'].extend(annotations)
                    info_data.append(image_stats)

        # 수정된 데이터 저장
        with open(anno_path, 'w') as f:
            json.dump(anno_data, f)
        
        with open(info_path, 'w') as f:
            json.dump(info_data, f)

        print(f"Validation and fixing of {subset} dataset completed.")
        print(f"Final image count: {len(anno_data['images'])}")

if __name__ == "__main__":
    output_dir = 'rtmdet_dataset2'
    target_counts = {
        'train': 32000,  # train 폴더 목표 이미지 수
        'val': 8000     # val 폴더 목표 이미지 수
    }
    validate_and_fix_dataset(output_dir, target_counts)

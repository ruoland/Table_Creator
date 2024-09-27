import json
import os
import torch
import torchvision.transforms.v2 as T
from torchvision.io import read_image, write_png
from torchvision.utils import draw_bounding_boxes
import torchvision.ops as ops

# JSON 파일 로드
json_file_path = r"C:\project\table2\train_annotations.json"
with open(json_file_path, 'r', encoding='utf-8') as file:
    annotations = json.load(file)

# 이미지 디렉토리 설정
image_dir = os.path.dirname(json_file_path)

# 변환 정의 (RandomPerspective만 사용)
transform = T.Compose([
    T.RandomPerspective(distortion_scale=0.5, p=1.0),
])

def process_image(image_info, image_annotations):
    # 이미지 로드
    image_path = os.path.join(image_dir, image_info['file_name'])
    image = read_image(image_path)

    # 바운딩 박스 추출 및 형식 변환
    bboxes = torch.tensor([ann['bbox'] for ann in image_annotations], dtype=torch.float32)
    bboxes = ops.box_convert(bboxes, 'xywh', 'xyxy')  # [x, y, w, h] to [x1, y1, x2, y2]
    
    # 원본 이미지에 원본 바운딩 박스 그리기
    original_result = draw_bounding_boxes(image, bboxes, width=3, colors="blue")
    
    # 이미지와 바운딩 박스에 변환 적용
    transformed = transform({"image": image, "boxes": bboxes})
    transformed_image = transformed["image"]
    transformed_bboxes = transformed["boxes"]
    # 변환된 이미지에 변환된 바운딩 박스 그리기
    transformed_result = draw_bounding_boxes(transformed_image, transformed_bboxes, width=3, colors="red")

    # 결과 저장
    write_png(original_result, os.path.join(image_dir, f"original_{image_info['file_name']}"))
    write_png(transformed_result, os.path.join(image_dir, f"transformed_{image_info['file_name']}"))

    # 변환된 바운딩 박스를 다시 원래 형식(xywh)으로 변환
    transformed_bboxes = ops.box_convert(transformed_bboxes, 'xyxy', 'xywh')
    for i, bbox in enumerate(transformed_bboxes):
        image_annotations[i]['bbox'] = bbox.tolist()

    return image_annotations

# 이미지별로 처리
for image in annotations['images']:
    image_id = image['id']
    image_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
    
    updated_annotations = process_image(image, image_annotations)
    
    # 원본 어노테이션 업데이트
    for ann in updated_annotations:
        idx = next(i for i, a in enumerate(annotations['annotations']) if a['id'] == ann['id'])
        annotations['annotations'][idx] = ann

# 변환된 어노테이션 저장
output_json_path = os.path.join(image_dir, 'transformed_annotations.json')
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(annotations, f, indent=2, ensure_ascii=False)

print(f"처리 완료. 변환된 어노테이션이 {output_json_path}에 저장되었습니다.")

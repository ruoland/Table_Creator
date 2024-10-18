import ujson as json
from collections import defaultdict

def analyze_coco_annotations(file_path):
    # JSON 파일 로드
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 이미지 크기 정보 저장
    image_sizes = {img['id']: (img['width'], img['height']) for img in data['images']}

    # 어노테이션 분석
    annotations = defaultdict(list)
    
    # RTMDet 기준 중간 객체 범위
    lower_bound = 0.03**2  # 0.0009
    upper_bound = 0.09**2  # 0.0081

    print(f"RTMDet 중간 크기 객체 범위: {lower_bound:.4f} ~ {upper_bound:.4f} (상대적 면적)")

    for ann in data['annotations']:
        category = ann['category_id']
        bbox = ann['bbox']
        area = bbox[2] * bbox[3]  # width * height
        image_width, image_height = image_sizes[ann['image_id']]
        
        # RTMDet에서 사용하는 상대적 크기 계산
        relative_area = area / (image_width * image_height)
        
        annotations[category].append((ann, relative_area))

    # 카테고리별 중간 크기 객체 분석
    for category, anns in annotations.items():
        medium_objects = [ann for ann, area in anns if lower_bound <= area <= upper_bound]
        
        print(f"\n카테고리 {category}:")
        print(f"  총 객체 수: {len(anns)}")
        print(f"  중간 크기 객체 수: {len(medium_objects)}")
        
        if medium_objects:
            print("  중간 크기 객체 샘플:")
            for i, ann in enumerate(medium_objects[:3]):  # 최대 3개 샘플 출력
                bbox = ann['bbox']
                area = bbox[2] * bbox[3] / (image_sizes[ann['image_id']][0] * image_sizes[ann['image_id']][1])
                print(f"    객체 {i+1}: bbox={bbox}, 상대적 면적={area:.6f}, 절대 면적={bbox[2]*bbox[3]}")

# 파일 경로 지정 및 함수 실행
file_path = r"C:\project\merged_dataset_5\train_annotations.json" # 실제 파일 경로로 변경하세요
analyze_coco_annotations(file_path)

import os
import shutil
import ujson as json
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def process_subset(args):
    dataset1_path, dataset2_path, output_path, subset, id_offset = args
    
    # 어노테이션 파일 로드
    with open(os.path.join(dataset1_path, f'{subset}_annotations.json'), 'r') as f:
        annotations1 = json.load(f)
    with open(os.path.join(dataset2_path, f'{subset}_annotations.json'), 'r') as f:
        annotations2 = json.load(f)

    # dataset1 처리
    for img in annotations1['images']:
        shutil.copy(
            os.path.join(dataset1_path, subset, 'images', img['file_name']),
            os.path.join(output_path, subset, 'images', img['file_name'])
        )

    # dataset2 처리 및 ID 업데이트
    for img in annotations2['images']:
        old_id = img['id']
        new_id = old_id + id_offset
        img['id'] = new_id
        new_filename = f"{new_id:06d}.jpeg"
        img['file_name'] = new_filename
        
        shutil.copy(
            os.path.join(dataset2_path, subset, 'images', f"{old_id:06d}.jpeg"),
            os.path.join(output_path, subset, 'images', new_filename)
        )

    # dataset2의 어노테이션 ID 업데이트
    for ann in annotations2['annotations']:
        ann['id'] += len(annotations1['annotations'])
        ann['image_id'] += id_offset

    # 어노테이션 병합
    merged_annotations = {
        'images': annotations1['images'] + annotations2['images'],
        'annotations': annotations1['annotations'] + annotations2['annotations'],
        'categories': annotations1['categories']
    }

    # 병합된 어노테이션 저장
    with open(os.path.join(output_path, f'{subset}_annotations.json'), 'w') as f:
        json.dump(merged_annotations, f)

    return f"Processed {subset} subset"

def merge_datasets(dataset1_path, dataset2_path, output_path):
    # 출력 디렉토리 생성
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'val', 'images'), exist_ok=True)

    # 이미지 ID 오프셋 계산 (dataset2의 이미지 ID를 위해)
    with open(os.path.join(dataset1_path, 'train_annotations.json'), 'r') as f:
        annotations1 = json.load(f)
    id_offset = max(img['id'] for img in annotations1['images']) + 1

    # 멀티프로세싱 설정
    num_processes = min(cpu_count(), 2)  # 최대 2개의 프로세스 사용 (train과 val)
    pool = Pool(processes=num_processes)

    # 작업 정의
    tasks = [
        (dataset1_path, dataset2_path, output_path, 'train', id_offset),
        (dataset1_path, dataset2_path, output_path, 'val', id_offset)
    ]

    # 멀티프로세싱 실행
    results = list(tqdm(pool.imap(process_subset, tasks), total=len(tasks), desc="Merging datasets"))

    pool.close()
    pool.join()

    print("Dataset merging completed.")
    for result in results:
        print(result)

if __name__ == "__main__":
    dataset1_path = r"C:\project\table_color2"
    dataset2_path = r"C:\project\table_color"
    output_path = r"C:\project\merged_1008"
    
    merge_datasets(dataset1_path, dataset2_path, output_path)

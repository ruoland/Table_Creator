import os
import json
import random
import shutil

def split_coco_annotations(base_dir, train_ratio=0.8):
    coco_file = os.path.join(base_dir, 'coco_annotations.json')
    with open(coco_file, 'r') as f:
        data = json.load(f)
    
    images = data['images']
    random.shuffle(images)
    split_index = int(len(images) * train_ratio)
    
    train_images = images[:split_index]
    val_images = images[split_index:]
    
    train_image_ids = set(img['id'] for img in train_images)
    val_image_ids = set(img['id'] for img in val_images)
    
    train_annotations = [ann for ann in data['annotations'] if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in data['annotations'] if ann['image_id'] in val_image_ids]
    
    train_data = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': data['categories']
    }
    
    val_data = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': data['categories']
    }
    
    ann_dir = os.path.join(base_dir, 'annotations')
    os.makedirs(ann_dir, exist_ok=True)
    
    with open(os.path.join(ann_dir, 'train.json'), 'w') as f:
        json.dump(train_data, f)
    
    with open(os.path.join(ann_dir, 'val.json'), 'w') as f:
        json.dump(val_data, f)
    
    print(f"Split annotations: {len(train_images)} train, {len(val_images)} val")

    # 이미지 파일 이동
    for subset, images in [('train', train_images), ('val', val_images)]:
        subset_dir = os.path.join(base_dir, subset, 'images')
        os.makedirs(subset_dir, exist_ok=True)
        for img in images:
            src = os.path.join(base_dir, 'train', 'images', img['file_name'])
            dst = os.path.join(subset_dir, img['file_name'])
            if os.path.exists(src):
                shutil.move(src, dst)

    print("Image files moved to respective directories")

def check_dataset_structure(base_dir):
    for subset in ['train', 'val']:
        print(f"\nChecking {subset} dataset:")
        
        images_dir = os.path.join(base_dir, subset, 'images')
        print(f"Images directory exists: {os.path.exists(images_dir)}")
        if os.path.exists(images_dir):
            print(f"Number of images: {len(os.listdir(images_dir))}")
        
        labels_dir = os.path.join(base_dir, subset, 'labels')
        print(f"Labels directory exists: {os.path.exists(labels_dir)}")
        if os.path.exists(labels_dir):
            print(f"Number of label files: {len(os.listdir(labels_dir))}")
        
        ann_file = os.path.join(base_dir, 'annotations', f'{subset}.json')
        print(f"Annotation file exists: {os.path.exists(ann_file)}")
        
        if os.path.exists(ann_file):
            with open(ann_file, 'r') as f:
                ann_data = json.load(f)
            print(f"Number of images in annotations: {len(ann_data['images'])}")
            print(f"Number of annotations: {len(ann_data['annotations'])}")
            
            print("\nSample image info:")
            print(json.dumps(ann_data['images'][0], indent=2))
            
            print("\nSample annotation info:")
            print(json.dumps(ann_data['annotations'][0], indent=2))
def check_annotation_image_consistency(base_dir, subset):
    ann_file = os.path.join(base_dir, 'annotations', f'{subset}.json')
    images_dir = os.path.join(base_dir, subset, 'images')

    with open(ann_file, 'r') as f:
        ann_data = json.load(f)

    ann_image_ids = set(img['id'] for img in ann_data['images'])
    actual_image_files = set(os.path.splitext(f)[0].split('_')[1] for f in os.listdir(images_dir) if f.endswith('.png'))

    missing_in_annotations = actual_image_files - ann_image_ids
    missing_in_directory = ann_image_ids - actual_image_files

    print(f"\nChecking consistency for {subset} dataset:")
    print(f"Images missing in annotations: {missing_in_annotations}")
    print(f"Images missing in directory: {missing_in_directory}")

if __name__ == "__main__":
    base_dir = 'D:/Projects/OCR-LEARNIGN-PROJECT/OCR-PROJECT_OLD/table_dataset_real'
    
    # 어노테이션 파일 분할
    split_coco_annotations(base_dir)
    
    print("Checking dataset structure:")
    check_dataset_structure(base_dir)
    check_annotation_image_consistency(base_dir, 'train')
    check_annotation_image_consistency(base_dir, 'val')
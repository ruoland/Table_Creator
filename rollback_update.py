import json

def print_annotation_structure(annotation_file):
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    print("Keys in annotation file:", data.keys())
    if 'images' in data:
        print("\nSample image entry:")
        print(json.dumps(data['images'][0], indent=2))
    
    if 'annotations' in data:
        print("\nSample annotation entry:")
        print(json.dumps(data['annotations'][0], indent=2))

# 훈련 세트와 검증 세트 모두에 대해 실행
print("Train annotations:")
print_annotation_structure(r'table_dataset_real\annotations\train.json')
print("\nValidation annotations:")
print_annotation_structure(r'table_dataset_real\annotations\val.json')

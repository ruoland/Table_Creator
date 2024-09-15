import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_table_structure(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    for image in data['images']:
        image_id = image['id']
        width = image['width']
        height = image['height']

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_title(f"Table Structure - Image ID: {image_id}")

        # 객체별 색상 설정
        colors = {'table': 'black', 'row': 'red', 'column': 'blue', 'cell': 'green', 'merged_cell': 'purple'}

        # 객체 간 관계 저장
        table_structure = {'rows': [], 'columns': [], 'cells': [], 'merged_cells': []}

        for ann in data['annotations']:
            if ann['image_id'] == image_id:
                category = categories[ann['category_id']]
                bbox = ann['bbox']
                
                if category == 'table':
                    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                             linewidth=2, edgecolor=colors[category], facecolor='none')
                    ax.add_patch(rect)
                    ax.text(bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2, category, 
                            color=colors[category], fontsize=10, ha='center', va='center')
                    
                elif category == 'row':
                    ax.axhline(y=bbox[1], color=colors[category], linestyle='--')
                    ax.text(bbox[0] + bbox[2]/2, bbox[1], f"{category} {ann['id']}", 
                            color=colors[category], fontsize=8, ha='center', va='bottom')
                
                elif category == 'column':
                    ax.axvline(x=bbox[0], color=colors[category], linestyle='--')
                    ax.text(bbox[0], bbox[1] + bbox[3]/2, f"{category} {ann['id']}", 
                            color=colors[category], fontsize=8, ha='right', va='center', rotation=90)
                
                elif category in ['cell', 'merged_cell']:
                    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                             linewidth=1, edgecolor=colors[category], facecolor='none')
                    ax.add_patch(rect)
                    ax.text(bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2, f"{category} {ann['id']}", 
                            color=colors[category], fontsize=6, ha='center', va='center')

        plt.tight_layout()
        plt.show()

# 사용 예:


# 사용 예:
visualize_table_structure(r'table_dataset_real\train_annotations.json')



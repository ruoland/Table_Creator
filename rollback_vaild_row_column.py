import json

def analyze_table_structure(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    for image in data['images']:
        image_id = image['id']
        print(f"\nAnalyzing Image ID: {image_id}")
        
        table_info = {'rows': [], 'columns': [], 'cells': [], 'merged_cells': []}
        
        for ann in data['annotations']:
            if ann['image_id'] == image_id:
                category = categories[ann['category_id']]
                
                if category == 'table':
                    if 'rows' in ann:
                        table_info['rows'] = ann['rows']
                    if 'columns' in ann:
                        table_info['columns'] = ann['columns']
                elif category == 'row':
                    table_info['rows'].append(ann['id'])
                elif category == 'column':
                    table_info['columns'].append(ann['id'])
                elif category == 'cell':
                    table_info['cells'].append(ann['id'])
                elif category == 'merged_cell':
                    table_info['merged_cells'].append(ann['id'])
        
        print(f"  Rows: {len(table_info['rows'])} ({table_info['rows']})")
        print(f"  Columns: {len(table_info['columns'])} ({table_info['columns']})")
        print(f"  Cells: {len(table_info['cells'])}")
        print(f"  Merged Cells: {len(table_info['merged_cells'])}")
        
        # 일관성 검사
        if not table_info['rows'] and not table_info['columns']:
            print("  WARNING: No rows or columns defined for this table.")
        if not table_info['cells'] and not table_info['merged_cells']:
            print("  WARNING: No cells defined for this table.")
        if len(table_info['rows']) * len(table_info['columns']) != len(table_info['cells']) + len(table_info['merged_cells']):
            print("  WARNING: Number of cells doesn't match rows * columns.")

# 사용 예:
analyze_table_structure(r'table_dataset_real\train_annotations.json')

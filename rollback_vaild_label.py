import json
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# 클래스별 색상 정의
class_colors = {
    1: 'cyan',     # Cell
    2: 'magenta',  # Merged Cell
    3: 'yellow',   # Row
    4: 'green',    # Column
    5: 'blue'      # Table
}

class_names = {
    1: 'cell',
    2: 'merged_cell',
    3: 'row',
    4: 'column',
    5: 'table'
}

def load_image_and_annotations(image_path, ann_file):
    img = Image.open(image_path)
    img_array = np.array(img)
    with open(ann_file, 'r') as f:
        ann_data = json.load(f)
    return img_array, ann_data
def plot_bounding_boxes(ax, annotations, image_id, selected_class, ann_file):
    ax.clear()
    
    img_info = next(img for img in annotations['images'] if img['id'] == image_id)
    img_path = os.path.join(os.path.dirname(ann_file), img_info['file_name'])
    img = Image.open(img_path)
    ax.imshow(img)

    legend_patches = []
    for class_id, class_name in class_names.items():
        if selected_class is None or class_id == selected_class:
            color = class_colors[class_id]
            legend_patches.append(patches.Patch(color=color, label=class_name))

    visible_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
    if not visible_annotations:
        ax.set_title(f"Image ID: {image_id} - No annotations found!")
        return

    for ann in visible_annotations:
        if selected_class is None or ann['category_id'] == selected_class:
            bbox = ann['bbox']
            category_id = ann['category_id']
            color = class_colors[category_id]
            
            if category_id in [1, 2]:  # Cell or Merged Cell
                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                         linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                label = 'Merged Cell' if category_id == 2 else 'Cell'
            elif category_id == 3:  # Row
                ax.axhline(y=bbox[1], xmin=bbox[0]/img_info['width'], xmax=(bbox[0]+bbox[2])/img_info['width'], 
                           color=color, linewidth=2, linestyle='--')
                label = 'Row'
            elif category_id == 4:  # Column
                ax.axvline(x=bbox[0], ymin=bbox[1]/img_info['height'], ymax=(bbox[1]+bbox[3])/img_info['height'], 
                           color=color, linewidth=2, linestyle='--')
                label = 'Column'
            elif category_id == 5:  # Table
                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                         linewidth=3, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                label = 'Table'

            # 레이블 추가
            ax.text(bbox[0], bbox[1], label, color=color, fontsize=8, 
                    verticalalignment='top', backgroundcolor='white')

    ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1, 1), 
              fontsize='small', fancybox=True, framealpha=0.7)
    ax.set_title(f"Image ID: {image_id}, Class: {class_names.get(selected_class, 'All')}")
    plt.draw()
def plot_rows_and_columns(annotations, image_id):
    img_info = next((img for img in annotations['images'] if img['id'] == image_id), None)
    if img_info is None:
        print(f"No image info found for image ID: {image_id}")
        return

    img_width, img_height = img_info['width'], img_info['height']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    ax1.set_title("Rows")
    ax1.set_xlim(0, img_width)
    ax1.set_ylim(img_height, 0)  # y축 반전
    
    ax2.set_title("Columns")
    ax2.set_xlim(0, img_width)
    ax2.set_ylim(img_height, 0)  # y축 반전

    print(f"Image ID: {image_id}, Width: {img_width}, Height: {img_height}")
    
    rows = []
    columns = []
    cells = []
    for ann in annotations['annotations']:
        if ann['image_id'] == image_id:
            if ann['category_id'] == 3:  # Row
                rows.append(ann)
            elif ann['category_id'] == 4:  # Column
                columns.append(ann)
            elif ann['category_id'] in [1, 2]:  # Cell or Merged Cell
                cells.append(ann)

    print(f"Number of rows: {len(rows)}")
    print(f"Number of columns: {len(columns)}")
    print(f"Number of cells: {len(cells)}")

    print("Rows:")
    for row in rows:
        bbox = row['bbox']
        print(f"  Row ID: {row['id']}, Y: {bbox[1]}, Height: {bbox[3]}")
        rect = patches.Rectangle((0, bbox[1]), img_width, bbox[3], 
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)
        ax1.text(0, bbox[1], f"Row {row['id']}", fontsize=8, 
                 verticalalignment='bottom')

    print("Columns:")
    for column in columns:
        bbox = column['bbox']
        print(f"  Column ID: {column['id']}, X: {bbox[0]}, Width: {bbox[2]}")
        rect = patches.Rectangle((bbox[0], 0), bbox[2], img_height, 
                                 linewidth=2, edgecolor='b', facecolor='none')
        ax2.add_patch(rect)
        ax2.text(bbox[0], 0, f"Column {column['id']}", fontsize=8, 
                 verticalalignment='bottom', rotation=90)

    print("Cells:")
    for cell in cells:
        bbox = cell['bbox']
        print(f"  Cell ID: {cell['id']}, X: {bbox[0]}, Y: {bbox[1]}, Width: {bbox[2]}, Height: {bbox[3]}")

    plt.tight_layout()
    plt.show()
def on_key(event, fig, ax, annotations, image_ids, current_index, selected_class, ann_file):
    if event.key == 'right':
        current_index[0] = (current_index[0] + 1) % len(image_ids)
    elif event.key == 'left':
        current_index[0] = (current_index[0] - 1) % len(image_ids)
    elif event.key in ['0', '1', '2', '3', '4', '5']:
        selected_class[0] = int(event.key) if event.key != '0' else None

    plot_bounding_boxes(ax, annotations, image_ids[current_index[0]], selected_class[0], ann_file)
    fig.canvas.draw()
def visualize_dataset(ann_file):
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    
    # 이미지 ID를 10개로 제한
    image_ids = sorted(set(ann['image_id'] for ann in annotations['annotations']))[:10]
    print(f"Available image IDs: {image_ids}")
    
    current_index = [0]
    selected_class = [None]

    fig, ax = plt.subplots(figsize=(12, 8))
    
    def update_plots():
        current_image_id = image_ids[current_index[0]]
        print(f"\nCurrent Image ID: {current_image_id}")
        plot_bounding_boxes(ax, annotations, current_image_id, selected_class[0], ann_file)
        fig.canvas.draw_idle()
        plot_rows_and_columns(annotations, current_image_id)

    update_plots()

    def on_key(event):
        if event.key == 'right':
            current_index[0] = (current_index[0] + 1) % len(image_ids)
        elif event.key == 'left':
            current_index[0] = (current_index[0] - 1) % len(image_ids)
        elif event.key in ['0', '1', '2', '3', '4', '5']:
            selected_class[0] = int(event.key) if event.key != '0' else None
        print(f"\nMoving to Image ID: {image_ids[current_index[0]]}")
        update_plots()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    def on_key(event):
        if event.key == 'right':
            current_index[0] = (current_index[0] + 1) % len(image_ids)
        elif event.key == 'left':
            current_index[0] = (current_index[0] - 1) % len(image_ids)
        elif event.key in ['0', '1', '2', '3', '4', '5']:
            selected_class[0] = int(event.key) if event.key != '0' else None
        update_plots()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

if __name__ == "__main__":
    ann_file = r"table_dataset_real\train_annotations.json"
    visualize_dataset(ann_file)
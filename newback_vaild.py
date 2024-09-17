import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
from collections import defaultdict

class DatasetVisualizer:
    def __init__(self, dataset_dir, subset='train'):
        self.dataset_dir = dataset_dir
        self.subset = subset
        self.load_dataset()
        self.current_image_id = None
        self.annotations = []
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.toggle = None
        self.setup_plot()

    def load_dataset(self):
        with open(os.path.join(self.dataset_dir, f'{self.subset}_annotations.json'), 'r') as f:
            self.coco_data = json.load(f)
        self.images = self.coco_data['images']
        self.category_names = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        with open(os.path.join(self.dataset_dir, 'dataset_info.json'), 'r') as f:
            self.dataset_info = json.load(f)
            
    def setup_plot(self):
        plt.subplots_adjust(bottom=0.2)
        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(self.next_image)
        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(self.prev_image)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_key(self, event):
        if event.key == 'right':
            self.next_image(event)
        elif event.key == 'left':
            self.prev_image(event)
        elif self.toggle:
            self.toggle.on_key(event)

    def next_image(self, event):
        if self.current_image_id is None:
            self.current_image_id = 0
        else:
            self.current_image_id = (self.current_image_id + 1) % len(self.images)
        self.show_image()

    def prev_image(self, event):
        if self.current_image_id is None:
            self.current_image_id = len(self.images) - 1
        else:
            self.current_image_id = (self.current_image_id - 1) % len(self.images)
        self.show_image()

    def show_image(self):
        self.ax.clear()
        image_info = self.images[self.current_image_id]
        image_path = os.path.join(self.dataset_dir, self.subset, 'images', image_info['file_name'])
        img = Image.open(image_path)
        self.ax.imshow(img, cmap='gray')

        self.annotations = [ann for ann in self.coco_data['annotations'] if ann['image_id'] == image_info['id']]
        lines = []
        text_annotations = []

        colors = {'normal_cell': 'r', 'merged_cell': 'g', 'row': 'b', 'column': 'c', 'table': 'm'}

        for ann in self.annotations:
            category = self.category_names[ann['category_id']]
            bbox = ann['bbox']
            color = colors[category]

            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                 fill=False, edgecolor=color, linewidth=2)
            self.ax.add_patch(rect)
            lines.append(rect)

            text_ann = self.ax.text(bbox[0], bbox[1], category, color=color,
                                    fontsize=8, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            text_annotations.append(text_ann)

        self.ax.set_title(f"Image ID: {image_info['id']} - Press n/m/r/c/t to toggle classes, 'a' to toggle all")
        plt.axis('off')

        self.toggle = ToggleVisibility(self.fig, self.ax, text_annotations, lines, self.annotations, self.category_names)
        self.fig.canvas.draw()

        # Print annotation counts
        class_counter = {category: sum(1 for ann in self.annotations if self.category_names[ann['category_id']] == category)
                         for category in colors.keys()}
        print(f"\nImage ID: {image_info['id']}")
        for category, count in class_counter.items():
            print(f"{category}: {count}")
        print(f"Total annotations: {len(self.annotations)}")
        # 검증 수행
        validation_results = self.validate_annotations(image_info['id'])
        for result in validation_results:
            print(result)
    def visualize(self):
        self.next_image(None)  # Show the first image
        plt.show()
    def validate_annotations(self, image_id):
        results = []
        image_info = next(info for info in self.dataset_info[self.subset] if info['image_id'] == image_id)
        annotations = [ann for ann in self.coco_data['annotations'] if ann['image_id'] == image_id]

        # 1. 병합된 셀 검증
        merged_cells = [ann for ann in annotations if self.category_names[ann['category_id']] == 'merged_cell']
        if len(merged_cells) != image_info['num_merged_cells']:
            results.append(f"Warning: Mismatch in number of merged cells. Expected: {image_info['num_merged_cells']}, Found: {len(merged_cells)}")

        # 2. 행과 열의 일관성 검증
        rows = [ann for ann in annotations if self.category_names[ann['category_id']] == 'row']
        cols = [ann for ann in annotations if self.category_names[ann['category_id']] == 'column']
        if len(rows) != image_info['rows'] or len(cols) != image_info['cols']:
            results.append(f"Warning: Mismatch in number of rows or columns. Expected: {image_info['rows']}x{image_info['cols']}, Found: {len(rows)}x{len(cols)}")

        # 3. 테이블 구조 검증
        table = [ann for ann in annotations if self.category_names[ann['category_id']] == 'table']
        if len(table) != 1:
            results.append(f"Error: Invalid number of table annotations. Expected: 1, Found: {len(table)}")

        # 4. 경계 상자 검증
        for ann in annotations:
            bbox = ann['bbox']
            if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] > image_info['image_width'] or bbox[1] + bbox[3] > image_info['image_height']:
                results.append(f"Error: Bounding box out of image bounds for {self.category_names[ann['category_id']]}")

        if not results:
            results.append("All validations passed successfully.")

        return results

class ToggleVisibility:
    def __init__(self, fig, ax, annotations, lines, raw_annotations, category_names):
        self.fig = fig
        self.ax = ax
        self.annotations = annotations
        self.lines = lines
        self.raw_annotations = raw_annotations
        self.category_names = category_names
        self.visible = defaultdict(lambda: True)

    def on_key(self, event):
        if event.key == 'n':
            self.toggle_class('normal_cell')
        elif event.key == 'm':
            self.toggle_class('merged_cell')
        elif event.key == 'r':
            self.toggle_class('row')
        elif event.key == 'c':
            self.toggle_class('column')
        elif event.key == 't':
            self.toggle_class('table')
        elif event.key == 'a':
            self.toggle_all()

    def toggle_class(self, class_name):
        self.visible[class_name] = not self.visible[class_name]
        for ann, line, raw_ann in zip(self.annotations, self.lines, self.raw_annotations):
            category = self.category_names[raw_ann['category_id']]
            if category == class_name:
                ann.set_visible(self.visible[class_name])
                line.set_visible(self.visible[class_name])
        self.fig.canvas.draw()

    def toggle_all(self):
        all_visible = all(self.visible.values())
        for class_name in self.visible:
            self.visible[class_name] = not all_visible
        for ann, line in zip(self.annotations, self.lines):
            ann.set_visible(not all_visible)
            line.set_visible(not all_visible)
        self.fig.canvas.draw()


if __name__ == "__main__":
    dataset_dir = 'yolox_table_dataset'
    visualizer = DatasetVisualizer(dataset_dir, subset='train')
    visualizer.visualize()
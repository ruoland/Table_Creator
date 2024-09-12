import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def test_image_and_labels(dataset_dir, num_samples=5):
    image_dir = os.path.join(dataset_dir, 'images')
    label_dir = os.path.join(dataset_dir, 'labels')
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    
    for _ in range(num_samples):
        image_file = random.choice(image_files)
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, image_file.replace('.png', '.txt'))
        
        # 이미지 로드
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # 레이블 로드
        with open(label_path, 'r') as f:
            labels = f.read().splitlines()
        
        # 그림 설정
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img, cmap='gray')
        
        # 레이블 그리기
        for label in labels:
            parts = label.split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
            
            # YOLO 형식을 픽셀 좌표로 변환
            x = (x_center - width/2) * img_width
            y = (y_center - height/2) * img_height
            w = width * img_width
            h = height * img_height
            
            # 박스 그리기
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
            if class_id == 0:  # 셀
                row, col = map(int, parts[5:7])
                ax.text(x, y, f'R{row}C{col}', color='r', fontsize=8, 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            elif class_id == 1:  # 전체 표
                rows, cols = map(int, parts[5:7])
                ax.text(x, y, f'Table {rows}x{cols}', color='b', fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        plt.title(f'Image: {image_file}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# 사용 예:
test_image_and_labels('table_dataset\\light', num_samples=5)

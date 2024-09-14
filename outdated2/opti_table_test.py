from ultralytics import YOLO
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

# 모델 로드
model = YOLO('epoch7.pt')

# 테스트 이미지 디렉토리
test_dir = r'table_dataset_new\test\images'

# 클래스 이름
class_names = ['cell', 'merged_cell', 'row', 'column', 'table']

# 색상 설정 (클래스별로 다른 색상)
colors = ['#%02x%02x%02x' % tuple(map(int, color)) for color in np.random.randint(0, 255, size=(len(class_names), 3))]

def draw_elements(image, boxes, class_name, color):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    
    for i, box in enumerate(boxes):
        cls = class_names[int(box.cls[0])]
        conf = float(box.conf[0])
        if conf > 0.5 and cls == class_name:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if class_name == 'row':
                draw.line([(x1, y1), (x2, y1)], fill=color, width=2)
            elif class_name == 'column':
                draw.line([(x1, y1), (x1, y2)], fill=color, width=2)
            else:  # cell or merged_cell
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            label = f'{i+1}:{cls} {conf:.2f}'
            bbox = draw.textbbox((x1, y1-10), label, font=font)
            draw.text((x1, y1-10), label, fill=color, font=font)
    
    return image

# 테스트 이미지에 대해 예측 수행
for img_name in os.listdir(test_dir):
    if img_name.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(test_dir, img_name)
        
        # 이미지 로드 및 그레이스케일 변환
        img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_3channel = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

        # 적응형 이진화 적용
        binary_img = cv2.adaptiveThreshold(
            gray_img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Block size
            5    # C constant
        )

        binary_3channel = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)

        # 예측 수행
        results = model(gray_3channel)
        
        # 원본 이미지를 PIL Image로 변환
        original_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # 각 요소별로 별도의 이미지 생성 및 저장
        for element in ['row', 'column', 'cell', 'merged_cell']:
            element_img = original_img.copy()
            element_img = draw_elements(element_img, results[0].boxes, element, colors[class_names.index(element)])
            element_img.save(f'results_{element}_{img_name}')
        
        print(f'Processed: {img_name}')

print("All images processed.")

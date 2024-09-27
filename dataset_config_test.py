import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

def create_table_image(width, height, rows, cols):
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    cell_width = width // cols
    cell_height = height // rows
    
    cells = []
    for row in range(rows):
        for col in range(cols):
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            
            draw.rectangle([x1, y1, x2, y2], outline="black")
            cells.append([x1, y1, x2, y2, row, col, False, False])
            
            font = ImageFont.load_default()
            text = f"Cell {row},{col}"
            text_bbox = draw.textbbox((x1, y1), text, font=font)
            text_position = ((x1 + x2 - text_bbox[2] + text_bbox[0]) // 2, 
                             (y1 + y2 - text_bbox[3] + text_bbox[1]) // 2)
            draw.text(text_position, text, fill="black", font=font)
    
    return np.array(image), cells

def apply_realistic_perspective(image, cells):
    height, width = image.shape[:2]
    
    # 더 현실적인 원근 변환을 위한 포인트 설정
    src_points = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    
    # 랜덤한 원근 효과 생성
    distortion = random.uniform(0.05, 0.2)
    tl_offset = random.uniform(0, distortion * width)
    tr_offset = random.uniform(0, distortion * width)
    bl_offset = random.uniform(0, distortion * width)
    br_offset = random.uniform(0, distortion * width)
    
    dst_points = np.float32([
        [tl_offset, random.uniform(0, distortion * height)],
        [width - 1 - tr_offset, random.uniform(0, distortion * height)],
        [width - 1 - br_offset, height - 1 - random.uniform(0, distortion * height)],
        [bl_offset, height - 1 - random.uniform(0, distortion * height)]
    ])
    
    # 변환 행렬 계산
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # 이미지에 원근 변환 적용
    warped_image = cv2.warpPerspective(image, matrix, (width, height))
    
    # 셀 좌표 변환
    transformed_cells = []
    for cell in cells:
        x1, y1, x2, y2 = cell[:4]
        points = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(points, matrix).reshape(-1, 2)
        
        new_points = []
        for point in transformed_points:
            new_x = max(0, min(point[0], width - 1))
            new_y = max(0, min(point[1], height - 1))
            new_points.append((new_x, new_y))
        
        transformed_cells.append(new_points + cell[4:])
    
    return warped_image, transformed_cells

def add_paper_texture(image):
    # 종이 텍스처 효과 추가
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    textured_image = cv2.add(image, noise)
    return textured_image
def add_shadow(image):
    # 그림자 효과 추가
    rows, cols = image.shape[:2]
    shadow = np.zeros((rows, cols), dtype=np.uint8)
    
    # 그림자 영역 생성 (예: 이미지의 왼쪽 상단 모서리)
    shadow_width = int(cols * 0.3)
    shadow_height = int(rows * 0.3)
    shadow[:shadow_height, :shadow_width] = 50  # 그림자 강도 조절 (0-255)
    
    # 그림자 블러 처리
    shadow = cv2.GaussianBlur(shadow, (21, 21), 0)
    
    # 그림자를 원본 이미지에 적용
    result = image.copy()
    for c in range(3):  # RGB 채널에 대해 반복
        result[:,:,c] = cv2.addWeighted(image[:,:,c], 1, shadow, 0.3, 0)
    
    return result
def visualize_cells_with_corners(image, cells, filename):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for cell in cells:
        corners = cell[:4]
        for i in range(4):
            start = corners[i]
            end = corners[(i+1) % 4]
            draw.line([start, end], fill="purple", width=2)
    
    img.save(filename)

def generate_realistic_table_image():
    width, height = random.randint(800, 1200), random.randint(600, 900)
    rows, cols = random.randint(3, 7), random.randint(3, 7)
    
    image, cells = create_table_image(width, height, rows, cols)
    
    # 원근 변환 적용
    warped_image, transformed_cells = apply_realistic_perspective(image, cells)
    
    # 종이 텍스처 효과 추가
    textured_image = add_paper_texture(warped_image)
    
    # 그림자 효과 추가
    final_image = add_shadow(textured_image)
    
    return final_image, transformed_cells

# 이미지 생성 및 저장
final_image, transformed_cells = generate_realistic_table_image()
visualize_cells_with_corners(final_image, transformed_cells, "realistic_table.png")
print("Realistic table image has been generated and saved as 'realistic_table.png'")

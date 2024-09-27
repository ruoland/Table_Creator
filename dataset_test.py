import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def apply_perspective(x, y, width, height, vanishing_point, strength):
    vx, vy = vanishing_point
    dx, dy = x - vx, y - vy
    distance = np.sqrt(dx**2 + dy**2)
    max_distance = np.sqrt(width**2 + height**2)
    scale = 1 - (distance / max_distance) * strength
    new_x = int(vx + dx * scale)
    new_y = int(vy + dy * scale)
    return new_x, new_y

def create_perspective_table(width, height, rows, cols, vanishing_point, strength):
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 20)
    
    cell_width = width // cols
    cell_height = height // rows
    
    cells = []
    
    for row in range(rows + 1):
        for col in range(cols + 1):
            x = col * cell_width
            y = row * cell_height
            
            new_x, new_y = apply_perspective(x, y, width, height, vanishing_point, strength)
            
            if col < cols and row < rows:
                next_x, next_y = apply_perspective(x + cell_width, y + cell_height, width, height, vanishing_point, strength)
                
                polygon = [
                    (new_x, new_y),
                    apply_perspective(x + cell_width, y, width, height, vanishing_point, strength),
                    (next_x, next_y),
                    apply_perspective(x, y + cell_height, width, height, vanishing_point, strength)
                ]
                
                draw.polygon(polygon, outline='black')
                
                cell_center = ((new_x + next_x) // 2, (new_y + next_y) // 2)
                text = f"R{row}C{col}"
                draw.text(cell_center, text, fill='black', font=font, anchor='mm')
                
                cells.append({
                    'row': row,
                    'col': col,
                    'bbox': [new_x, new_y, next_x, next_y]
                })
    
    return img, cells

# 랜덤 표 생성
width, height = 1000, 800
rows = random.randint(3, 7)
cols = random.randint(3, 7)

# 원근법 설정
vanishing_point = (width // 2, height // 2)  # 소실점을 이미지 중앙으로 설정
strength = 0.5  # 원근감 강도 (0.0 ~ 1.0)

img, cells = create_perspective_table(width, height, rows, cols, vanishing_point, strength)

# 결과 저장
img.save('perspective_table_whole.png')

# 레이블링 정보 출력
for cell in cells:
    print(f"Cell at row {cell['row']}, col {cell['col']}: bbox = {cell['bbox']}")

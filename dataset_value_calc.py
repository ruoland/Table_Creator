import cv2
import numpy as np
from PIL import Image, ImageDraw

def create_simple_table():
    # 간단한 3x3 표 생성
    img = Image.new('RGB', (300, 300), color='white')
    draw = ImageDraw.Draw(img)
    
    # 표 그리기
    for i in range(4):
        draw.line([(0, i*100), (300, i*100)], fill='black', width=2)
        draw.line([(i*100, 0), (i*100, 300)], fill='black', width=2)
    
    # 셀 좌표 저장
    cells = []
    for i in range(3):
        for j in range(3):
            cells.append([j*100, i*100, (j+1)*100, (i+1)*100])
    
    return np.array(img), cells

def apply_perspective(img, cells):
    rows, cols = img.shape[:2]
    
    # 원근 변환을 위한 점 설정
    src_points = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
    dst_points = np.float32([[0, 0], [cols-1, 0], [int(cols*0.1), rows-1], [int(cols*0.9), rows-1]])
    
    # 변환 행렬 계산
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # 이미지에 원근 변환 적용
    result = cv2.warpPerspective(img, matrix, (cols, rows))
    
    # 셀 좌표에 원근 변환 적용
    transformed_cells = []
    for cell in cells:
        points = np.float32([[cell[0], cell[1]], [cell[2], cell[1]], [cell[2], cell[3]], [cell[0], cell[3]]]).reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(points, matrix).reshape(-1, 2)
        x_min, y_min = np.min(transformed_points, axis=0)
        x_max, y_max = np.max(transformed_points, axis=0)
        transformed_cells.append([int(x_min), int(y_min), int(x_max), int(y_max)])
    
    return result, transformed_cells

def draw_cell_coordinates(img, cells):
    for i, cell in enumerate(cells):
        cv2.rectangle(img, (cell[0], cell[1]), (cell[2], cell[3]), (0, 255, 0), 2)
        cv2.putText(img, f"Cell {i+1}", (cell[0], cell[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return img

# 메인 실행
original_img, original_cells = create_simple_table()
transformed_img, transformed_cells = apply_perspective(original_img, original_cells)

# 원본 이미지에 셀 좌표 표시
original_with_coords = draw_cell_coordinates(original_img.copy(), original_cells)

# 변환된 이미지에 셀 좌표 표시
transformed_with_coords = draw_cell_coordinates(transformed_img.copy(), transformed_cells)

# 결과 저장
cv2.imwrite('original_table.png', cv2.cvtColor(original_with_coords, cv2.COLOR_RGB2BGR))
cv2.imwrite('transformed_table.png', transformed_with_coords)

print("Original cell coordinates:", original_cells)
print("Transformed cell coordinates:", transformed_cells)

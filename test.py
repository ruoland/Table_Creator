import cv2
import numpy as np
from PIL import Image, ImageDraw
import random
from PIL import ImageFont

def transform_cell_coordinates(cell, matrix, width, height):
    x1, y1, x2, y2 = cell[:4]
    points = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(points, matrix).reshape(-1, 2)
    
    new_x1 = max(0, min(transformed_points[:, 0].min(), width - 1))
    new_y1 = max(0, min(transformed_points[:, 1].min(), height - 1))
    new_x2 = max(0, min(transformed_points[:, 0].max(), width - 1))
    new_y2 = max(0, min(transformed_points[:, 1].max(), height - 1))
    
    return [new_x1, new_y1, new_x2, new_y2]  # cell[4:]를 제거

def create_grid_test_image_with_cells(width, height, grid_size=50, cell_count=5, perspective_matrix=None):
    # 원본 격자 이미지 생성
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)

    # 세로선 그리기
    for x in range(0, width, grid_size):
        draw.line([(x, 0), (x, height)], fill='lightgray', width=1)

    # 가로선 그리기
    for y in range(0, height, grid_size):
        draw.line([(0, y), (width, y)], fill='lightgray', width=1)

    # 격자 교차점에 작은 점 찍기
    for x in range(0, width, grid_size):
        for y in range(0, height, grid_size):
            draw.ellipse([(x-1, y-1), (x+1, y+1)], fill='gray')

    # 큰 셀 생성
    cells = []
    for _ in range(cell_count):
        cell_width = random.randint(grid_size*2, grid_size*5)
        cell_height = random.randint(grid_size*2, grid_size*5)
        x = random.randint(0, width - cell_width)
        y = random.randint(0, height - cell_height)
        cells.append((x, y, x + cell_width, y + cell_height))

    # 큰 셀 그리기
    for i, (x1, y1, x2, y2) in enumerate(cells):
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        draw.rectangle([x1, y1, x2, y2], outline='black', width=2, fill=color)
        # 셀 번호 표시
        draw.text((x1+5, y1+5), str(i+1), fill='black')

    # 원근 변환 적용 (만약 perspective_matrix가 제공된 경우)
    if perspective_matrix is not None:
        # PIL 이미지를 OpenCV 형식으로 변환
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 원근 변환 적용
        warped_image = cv2.warpPerspective(cv_image, perspective_matrix, (width, height))
        
        # 다시 PIL 이미지로 변환
        warped_pil = Image.fromarray(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
        
        return image, warped_pil, cells
    
    return image, None, cells

def get_perspective_points(width, height, intensity, direction=None):
    # 최대 오프셋을 이미지 크기의 일정 비율로 설정 (약 30도에 해당)
    max_offset_x = width * intensity
    max_offset_y = height * intensity
    
    if direction is None:
        # 랜덤 방향
        top_left_x = random.uniform(0, max_offset_x)
        top_left_y = random.uniform(0, max_offset_y)
        top_right_x = random.uniform(width - max_offset_x, width)
        top_right_y = random.uniform(0, max_offset_y)
        bottom_right_x = random.uniform(width - max_offset_x, width)
        bottom_right_y = random.uniform(height - max_offset_y, height)
        bottom_left_x = random.uniform(0, max_offset_x)
        bottom_left_y = random.uniform(height - max_offset_y, height)
    else:
        # 지정된 방향에 따른 변형
        if direction in ['left', 'top_left', 'bottom_left']:
            top_left_x = bottom_left_x = random.uniform(max_offset_x/2, max_offset_x)
        else:
            top_left_x = bottom_left_x = 0

        if direction in ['right', 'top_right', 'bottom_right']:
            top_right_x = bottom_right_x = random.uniform(width - max_offset_x, width - max_offset_x/2)
        else:
            top_right_x = bottom_right_x = width

        if direction in ['top', 'top_left', 'top_right']:
            top_left_y = top_right_y = random.uniform(max_offset_y/2, max_offset_y)
        else:
            top_left_y = top_right_y = 0

        if direction in ['bottom', 'bottom_left', 'bottom_right']:
            bottom_left_y = bottom_right_y = random.uniform(height - max_offset_y, height - max_offset_y/2)
        else:
            bottom_left_y = bottom_right_y = height

    points = np.float32([
        [top_left_x, top_left_y],
        [top_right_x, top_right_y],
        [bottom_right_x, bottom_right_y],
        [bottom_left_x, bottom_left_y]
    ])
    print("Perspective points:", points)  # 로그 추가
    return points
def create_grid_test_image_with_cells(width, height, grid_size=50, cell_count=5, perspective_matrix=None):
    # 원본 격자 이미지 생성
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)

    # 격자선 그리기
    for x in range(0, width, grid_size):
        draw.line([(x, 0), (x, height)], fill='black', width=1)
    for y in range(0, height, grid_size):
        draw.line([(0, y), (width, y)], fill='black', width=1)

    # 격자 교차점에 빨간 점 찍기
    for x in range(0, width, grid_size):
        for y in range(0, height, grid_size):
            draw.ellipse([(x-3, y-3), (x+3, y+3)], fill='red')

    # 큰 셀 생성
    cells = []
    for i in range(cell_count):
        cell_width = random.randint(grid_size*3, grid_size*5)
        cell_height = random.randint(grid_size*3, grid_size*5)
        x = random.randint(0, width - cell_width)
        y = random.randint(0, height - cell_height)
        cells.append((x, y, x + cell_width, y + cell_height))

    # 큰 셀 그리기
    for i, (x1, y1, x2, y2) in enumerate(cells):
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        draw.rectangle([x1, y1, x2, y2], outline='black', width=2, fill=color)
        # 셀 번호 표시
        draw.text((x1+5, y1+5), str(i+1), fill='black', font=ImageFont.truetype("arial.ttf", 20))

    # 대각선 그리기 (변형 효과를 더 잘 보기 위해)
    draw.line([(0, 0), (width, height)], fill='blue', width=2)
    draw.line([(width, 0), (0, height)], fill='blue', width=2)

    # 원근 변환 적용 (만약 perspective_matrix가 제공된 경우)
    if perspective_matrix is not None:
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        warped_image = cv2.warpPerspective(cv_image, perspective_matrix, (width, height))
        warped_pil = Image.fromarray(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
        return image, warped_pil, cells
    
    return image, None, cells

# PIL에서 폰트를 사용하기 위한 import 추가
def test_perspective_transform_with_cells(width, height, perspective_intensity):
    src_points = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    dst_points = get_perspective_points(width, height, perspective_intensity)
    print("Source points:", src_points)
    print("Destination points:", dst_points)

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    print("Perspective matrix:", matrix)

    original, warped, cells = create_grid_test_image_with_cells(width, height, grid_size=50, cell_count=5, perspective_matrix=matrix)

    original.save("original_grid_with_cells.png")
    warped.save("warped_grid_with_cells.png")

    print("Test images have been generated: 'original_grid_with_cells.png' and 'warped_grid_with_cells.png'")
    
    for i, cell in enumerate(cells):
        print(f"Original Cell {i+1}: {cell}")
        transformed_cell = transform_cell_coordinates(cell, matrix, width, height)
        print(f"Transformed Cell {i+1}: {transformed_cell}")
        print(f"Difference: {np.array(transformed_cell) - np.array(cell)}")
        print()

# 테스트 실행 (강도 증가)
test_perspective_transform_with_cells(800, 600, 0.3)  # 강도를 0.5로 증가
# 테스트 실행

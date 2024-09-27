from dataset_config import logging
from PIL import Image, ImageColor, ImageFont, ImageDraw, ImageFilter, ImageEnhance, ImageChops
import random
import cv2
import numpy as np

def add_noise(image, intensity=0.1):
    """
    이미지에 노이즈를 추가합니다.
    :param image: PIL Image 객체
    :param noise_type: 'gaussian'
    :param intensity: 노이즈 강도 (0.0 ~ 1.0)
    :return: 노이즈가 추가된 PIL Image 객체
    """
    width, height = image.size
    noise_image = image.copy()
    draw = ImageDraw.Draw(noise_image)

    for x in range(width):
        for y in range(height):
            r, g, b = image.getpixel((x, y))
            noise = random.gauss(0, 15 * intensity)
            r = max(0, min(255, int(r + noise)))
            g = max(0, min(255, int(g + noise)))
            b = max(0, min(255, int(b + noise)))
            draw.point((x, y), fill=(r, g, b))
    
    
    return noise_image
def apply_realistic_effects(image, cells, table_bbox, title_height, config):
    transform_matrix = np.eye(3)  # 기본값으로 항등 행렬 설정

    if config.enable_perspective_transform:
        image, cells, table_bbox, transform_matrix = apply_perspective_transform(
            image, 
            cells, 
            table_bbox, 
            title_height,
            config.perspective_intensity,
            config
        )
    

    else:
        transform_matrix = np.eye(3)
    
    # 새로운 이미지 크기 저장
    new_width, new_height = image.size
    if config.enable_noise:
        noise_intensity = random.uniform(*config.noise_intensity_range)
        image = add_noise(image, noise_intensity)
    
    if config.enable_blur:
        blur_radius = random.uniform(*config.blur_radius_range)
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    if config.enable_brightness_variation:
        brightness_factor = random.uniform(*config.brightness_factor_range)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
    
    if config.enable_contrast_variation:
        contrast_factor = random.uniform(*config.contrast_factor_range)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)

    if config.enable_shadow:
        shadow_opacity = config.shadow_opacity  # 직접 설정된 값 사용
        shadow_blur = config.shadow_blur_radius
        image = add_shadow(image, opacity=shadow_opacity, blur_radius=shadow_blur)    
    return image, cells, table_bbox, transform_matrix, new_width, new_height
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

    return np.float32([
        [top_left_x, top_left_y],
        [top_right_x, top_right_y],
        [bottom_right_x, bottom_right_y],
        [bottom_left_x, bottom_left_y]
    ])


def apply_perspective_transform(pil_image, cells, table_bbox, title_height, intensity, config):
    width, height = pil_image.size
    
    # 원근 변환을 위한 소스 포인트
    src_points = np.float32([[0, title_height], [width-1, title_height], 
                             [width-1, height-1], [0, height-1]])
    
    # 목표 포인트 계산 (랜덤한 변형 또는 지정된 방향으로 변형 적용)
    direction = getattr(config, 'perspective_direction', None)
    dst_points = get_perspective_points(width, height - title_height, intensity, direction)
    dst_points += np.float32([0, title_height])  # 타이틀 높이 고려
    
    # 변환 행렬 계산
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # 이미지에 원근 변환 적용
    opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    warped_image = cv2.warpPerspective(opencv_image, matrix, (width, height))
    
    # 셀과 테이블 좌표 변환
    transformed_cells = [transform_coordinates(cell, matrix, width, height) for cell in cells]
    transformed_table_bbox = transform_coordinates(table_bbox, matrix, width, height)
    
    # 최소 셀 크기 설정
    min_cell_size = 5  # 픽셀 단위
    transformed_cells = [
        cell if (cell[2] - cell[0] >= min_cell_size and cell[3] - cell[1] >= min_cell_size) else None
        for cell in transformed_cells
    ]
    transformed_cells = [cell for cell in transformed_cells if cell is not None]
    
    # PIL 이미지로 변환
    pil_warped = Image.fromarray(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
    
    # 그림자 추가 (config 사용)
    if config.enable_shadow:
        shadow_directions = get_shadow_direction(dst_points)
        pil_warped = add_directional_shadow(pil_warped, shadow_directions, config)
    transformed_cells = validate_labels(transformed_cells, width, height)
    log_transformation_info(cells, transformed_cells, intensity)

    return pil_warped, transformed_cells, transformed_table_bbox, matrix
def validate_labels(cells, image_width, image_height):
    validated_cells = []
    for cell in cells:
        x1, y1, x2, y2 = cell[:4]
        # 좌표가 이미지 범위 내에 있는지 확인
        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))
        # 셀의 너비와 높이가 최소값 이상인지 확인
        if x2 - x1 >= 1 and y2 - y1 >= 1:
            validated_cells.append([x1, y1, x2, y2] + cell[4:])
    return validated_cells

# apply_perspective_transform 함수 내에서 사용
def log_transformation_info(original_cells, transformed_cells, intensity):
    original_count = len(original_cells)
    transformed_count = len(transformed_cells)
    logging.info(f"Perspective transform applied with intensity {intensity}")
    logging.info(f"Original cell count: {original_count}")
    logging.info(f"Transformed cell count: {transformed_count}")
    if transformed_count < original_count:
        logging.warning(f"Lost {original_count - transformed_count} cells during transformation")

# apply_perspective_transform 함수 내에서 사용
def transform_coordinates(coords, matrix, width, height):
    x1, y1, x2, y2 = coords[:4]
    points = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(points, matrix).reshape(-1, 2)
    
    # 변환된 좌표가 이미지 경계 내에 있도록 보장
    transformed_points[:, 0] = np.clip(transformed_points[:, 0], 0, width - 1)
    transformed_points[:, 1] = np.clip(transformed_points[:, 1], 0, height - 1)
    
    new_x1 = np.min(transformed_points[:, 0])
    new_y1 = np.min(transformed_points[:, 1])
    new_x2 = np.max(transformed_points[:, 0])
    new_y2 = np.max(transformed_points[:, 1])
    
    return [new_x1, new_y1, new_x2, new_y2] + coords[4:]

def get_perspective_points(width, height, intensity, direction=None):
    # 최대 오프셋을 이미지 크기의 일정 비율로 설정 (약 15-20도에 해당)
    max_offset_x = width * intensity * 0.15
    max_offset_y = height * intensity * 0.15
    
    if direction is None:
        # 랜덤 방향 (기존 방식)
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
            top_left_x = random.uniform(max_offset_x/2, max_offset_x)
            bottom_left_x = random.uniform(max_offset_x/2, max_offset_x)
        else:
            top_left_x = bottom_left_x = 0

        if direction in ['right', 'top_right', 'bottom_right']:
            top_right_x = random.uniform(width - max_offset_x, width - max_offset_x/2)
            bottom_right_x = random.uniform(width - max_offset_x, width - max_offset_x/2)
        else:
            top_right_x = bottom_right_x = width

        if direction in ['top', 'top_left', 'top_right']:
            top_left_y = random.uniform(max_offset_y/2, max_offset_y)
            top_right_y = random.uniform(max_offset_y/2, max_offset_y)
        else:
            top_left_y = top_right_y = 0

        if direction in ['bottom', 'bottom_left', 'bottom_right']:
            bottom_left_y = random.uniform(height - max_offset_y, height - max_offset_y/2)
            bottom_right_y = random.uniform(height - max_offset_y, height - max_offset_y/2)
        else:
            bottom_left_y = bottom_right_y = height

    return np.float32([
        [top_left_x, top_left_y],
        [top_right_x, top_right_y],
        [bottom_right_x, bottom_right_y],
        [bottom_left_x, bottom_left_y]
    ])

def add_directional_shadow(image, directions, config):
    width, height = image.size
    shadow = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)
    
    shadow_size = int(min(width, height) * config.shadow_size_ratio)
    
    for direction in directions:
        if direction == 'left':
            for i in range(shadow_size):
                alpha = int((1 - i / shadow_size) * config.shadow_opacity)
                draw.line([(i, 0), (i, height)], fill=(0, 0, 0, alpha))
        elif direction == 'right':
            for i in range(shadow_size):
                alpha = int((1 - i / shadow_size) * config.shadow_opacity)
                draw.line([(width - 1 - i, 0), (width - 1 - i, height)], fill=(0, 0, 0, alpha))
        elif direction == 'top':
            for i in range(shadow_size):
                alpha = int((1 - i / shadow_size) * config.shadow_opacity)
                draw.line([(0, i), (width, i)], fill=(0, 0, 0, alpha))
        elif direction == 'bottom':
            for i in range(shadow_size):
                alpha = int((1 - i / shadow_size) * config.shadow_opacity)
                draw.line([(0, height - 1 - i), (width, height - 1 - i)], fill=(0, 0, 0, alpha))
    
    # 그림자 부드럽게 만들기
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=config.shadow_blur_radius))
    
    result = Image.alpha_composite(image.convert('RGBA'), shadow)
    return result.convert('RGB')


def transform_cell_coordinates(cell, matrix, margin_x, margin_y):
    x1, y1, x2, y2 = cell[:4]
    points = np.float32([[x1 + margin_x, y1 + margin_y],
                         [x2 + margin_x, y1 + margin_y],
                         [x2 + margin_x, y2 + margin_y],
                         [x1 + margin_x, y2 + margin_y]])
    transformed_points = cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    
    new_x1 = np.min(transformed_points[:, 0])
    new_y1 = np.min(transformed_points[:, 1])
    new_x2 = np.max(transformed_points[:, 0])
    new_y2 = np.max(transformed_points[:, 1])
    
    return [new_x1, new_y1, new_x2, new_y2] + cell[4:]

def adjust_coordinates(coords_list, crop_bbox):
    x_offset, y_offset = crop_bbox[0], crop_bbox[1]
    return [[c[0] - x_offset, c[1] - y_offset, c[2] - x_offset, c[3] - y_offset] + c[4:] for c in coords_list]

def transform_cell_coordinates(cell, matrix, margin_x, margin_y):
    x1, y1, x2, y2 = cell[:4]
    points = np.float32([[x1 + margin_x, y1 + margin_y],
                         [x2 + margin_x, y1 + margin_y],
                         [x2 + margin_x, y2 + margin_y],
                         [x1 + margin_x, y2 + margin_y]])
    transformed_points = cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    
    new_x1 = max(0, min(transformed_points[:, 0]))
    new_y1 = max(0, min(transformed_points[:, 1]))
    new_x2 = max(0, max(transformed_points[:, 0]))
    new_y2 = max(0, max(transformed_points[:, 1]))
    
    return [new_x1, new_y1, new_x2, new_y2] + cell[4:]

def adjust_cell_coordinates(cells, bbox):
    x_offset, y_offset = bbox[0], bbox[1]
    return [[c[0] - x_offset, c[1] - y_offset, c[2] - x_offset, c[3] - y_offset] + c[4:] for c in cells]

def get_average_color(image):
    return tuple(int(x) for x in np.array(image).mean(axis=(0,1)))

def find_content_bbox(image, min_margin_ratio=0.05):
    bg = Image.new(image.mode, image.size, image.getpixel((0,0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    
    if bbox:
        width, height = image.size
        min_margin = int(min(width, height) * min_margin_ratio)
        bbox = (max(bbox[0] - min_margin, 0),
                max(bbox[1] - min_margin, 0),
                min(bbox[2] + min_margin, width),
                min(bbox[3] + min_margin, height))
    else:
        bbox = (0, 0, image.width, image.height)
    
    return bbox

def add_shadow(image, opacity, blur_radius):
    width, height = image.size
    shadow = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)
    
    # 그림자 방향과 크기 랜덤 설정
    direction = random.choice(['left', 'right', 'top', 'bottom'])
    shadow_width = random.randint(10, 30)
    
    if direction == 'left':
        draw.rectangle([0, 0, shadow_width, height], fill=(0, 0, 0, opacity))
    elif direction == 'right':
        draw.rectangle([width - shadow_width, 0, width, height], fill=(0, 0, 0, opacity))
    elif direction == 'top':
        draw.rectangle([0, 0, width, shadow_width], fill=(0, 0, 0, opacity))
    else:  # bottom
        draw.rectangle([0, height - shadow_width, width, height], fill=(0, 0, 0, opacity))
    
    # 그림자 블러 처리
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # 원본 이미지와 그림자 합성
    result = Image.alpha_composite(image.convert('RGBA'), shadow)
    return result.convert('RGB')


def get_shadow_direction(dst_points):
    top_left, top_right, bottom_right, bottom_left = dst_points
    
    directions = []
    if top_left[1] > bottom_left[1]:
        directions.append('top')
    if top_right[1] > top_left[1]:
        directions.append('right')
    if bottom_right[0] < top_right[0]:
        directions.append('left')
    if bottom_right[1] > top_right[1]:
        directions.append('bottom')
    
    return directions if directions else ['bottom']  # 기본값으로 'bottom' 반환
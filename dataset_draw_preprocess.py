from logging_config import table_logger, get_memory_handler

from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageChops
import random
import numpy as np
from typing import List, Tuple, Optional, Dict
import random
from typing import List, Tuple
from PIL import Image, ImageEnhance, ImageFilter
# 상수 정의
MIN_CELL_SIZE = 5  # 픽셀 단위

def add_noise(image: Image.Image, intensity: float = 0.1) -> Image.Image:
    """
    이미지에 노이즈를 추가합니다.
    :param image: PIL Image 객체
    :param intensity: 노이즈 강도 (0.0 ~ 1.0)
    :return: 노이즈가 추가된 PIL Image 객체
    """
    np_image = np.array(image)
    noise = np.random.normal(0, 15 * intensity, np_image.shape).astype(np.int8)
    np_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(np_image)


def apply_effect(image: Image.Image, config, noise_intensity=None, blur_radius=None,
                 brightness_factor=None, contrast_factor=None) -> Image.Image:
    if config.enable_noise and noise_intensity is not None:
        return add_noise(image, noise_intensity)
    if config.enable_blur and blur_radius is not None:
        return image #image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    if config.enable_brightness_variation and brightness_factor is not None:
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(brightness_factor)
    if config.enable_contrast_variation and contrast_factor is not None:
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(contrast_factor)
    return image

def apply_realistic_effects(image: Image.Image, cells: List[List[float]], table_bbox: List[float], 
                            config) -> Tuple[Image.Image, List[List[float]], List[float], int, int]:
    table_logger.debug(f"Applying effects: perspective={config.enable_perspective_transform}, "
                       f"noise={config.enable_noise}, blur={config.enable_blur}")

    new_width, new_height = image.size

    # 표 잘라내기 기능 추가
    if random.random() < config.table_crop_probability:
        image, cells, table_bbox = crop_table(image, cells, table_bbox, config)

    # 그림자 추가
    if config.enable_shadow:
        all_directions = ['bottom', 'right', 'left', 'top']
        num_directions = random.randint(1, 4)
        shadow_directions = random.sample(all_directions, num_directions)
        shadow_opacity = random.randint(*config.shadow_opacity_range)
        image = add_directional_shadow(image, shadow_directions, config.shadow_blur_radius,
                                       config.shadow_size_ratio, shadow_opacity)

    # 랜덤 값 미리 생성
    noise_intensity = random.uniform(*config.noise_intensity_range) if config.enable_noise else None
    blur_radius = random.uniform(*config.blur_radius_range) if config.enable_blur else None
    brightness_factor = random.uniform(*config.brightness_factor_range) if config.enable_brightness_variation else None
    contrast_factor = random.uniform(*config.contrast_factor_range) if config.enable_contrast_variation else None

    # 효과 적용
    image = apply_effect(image, config, noise_intensity=noise_intensity,
                         blur_radius=blur_radius,
                         brightness_factor=brightness_factor,
                         contrast_factor=contrast_factor)

    return image, cells, table_bbox, new_width, new_height


def crop_table(image: Image.Image, cells: List[Dict], table_bbox: List[float], config) -> Tuple[Image.Image, List[Dict], List[float]]:
    # 병합된 셀이나 오버플로우 셀이 있는지 확인
    if any('is_merged' in cell and cell['is_merged'] for cell in cells) or \
       any('overflow' in cell and cell['overflow'] for cell in cells):
        table_logger.info("Table contains merged or overflow cells. Skipping crop operation.")
        return image, cells, table_bbox

    
    crop_direction = random.choice(['left', 'right', 'top', 'bottom'])
    crop_amount = random.uniform(0.1, 0.3)
    
    x1, y1, x2, y2 = table_bbox
    table_width = x2 - x1
    table_height = y2 - y1
    
    if crop_direction == 'left':
        crop_pixels = int(table_width * crop_amount)
        new_table_bbox = [x1 + crop_pixels, y1, x2, y2]
    elif crop_direction == 'right':
        crop_pixels = int(table_width * crop_amount)
        new_table_bbox = [x1, y1, x2 - crop_pixels, y2]
    elif crop_direction == 'top':
        crop_pixels = int(table_height * crop_amount)
        new_table_bbox = [x1, y1 + crop_pixels, x2, y2]
    else:  # bottom
        crop_pixels = int(table_height * crop_amount)
        new_table_bbox = [x1, y1, x2, y2 - crop_pixels]
    
    new_image = image.crop(new_table_bbox)
    
    new_cells = []
    for cell in cells:
        new_cell = cell.copy()
        
        # 기본 셀 좌표 조정
        new_cell['x1'] = max(0, min(cell['x1'] - new_table_bbox[0], new_image.width))
        new_cell['y1'] = max(0, min(cell['y1'] - new_table_bbox[1], new_image.height))
        new_cell['x2'] = max(0, min(cell['x2'] - new_table_bbox[0], new_image.width))
        new_cell['y2'] = max(0, min(cell['y2'] - new_table_bbox[1], new_image.height))
        
        # 유효한 셀만 포함 (완전히 잘린 셀은 제외)
        if new_cell['x2'] > new_cell['x1'] and new_cell['y2'] > new_cell['y1']:
            new_cells.append(new_cell)
    
    final_table_bbox = [0, 0, new_image.width, new_image.height]
    
    table_logger.info(f"Table cropped: direction={crop_direction}, amount={crop_amount:.2f}, new size: {new_image.width}x{new_image.height}")
    
    return new_image, new_cells, final_table_bbox



def add_directional_shadow(image: Image.Image, directions: List[str], blur_radius: float, 
                           size_ratio: float, opacity: int) -> Image.Image:
    """
    이미지에 방향성 그림자를 추가합니다.
    """
    width, height = image.size
    shadow = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)
    
    shadow_size = int(min(width, height) * size_ratio)
    
    for direction in directions:
        if direction == 'left':
            for i in range(shadow_size):
                alpha = int((1 - i / shadow_size) * opacity)
                draw.line([(i, 0), (i, height)], fill=(0, 0, 0, alpha))
        elif direction == 'right':
            for i in range(shadow_size):
                alpha = int((1 - i / shadow_size) * opacity)
                draw.line([(width - 1 - i, 0), (width - 1 - i, height)], fill=(0, 0, 0, alpha))
        elif direction == 'top':
            for i in range(shadow_size):
                alpha = int((1 - i / shadow_size) * opacity)
                draw.line([(0, i), (width, i)], fill=(0, 0, 0, alpha))
        elif direction == 'bottom':
            for i in range(shadow_size):
                alpha = int((1 - i / shadow_size) * opacity)
                draw.line([(0, height - 1 - i), (width, height - 1 - i)], fill=(0, 0, 0, alpha))
    
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    result = Image.alpha_composite(image.convert('RGBA'), shadow)
    return result.convert('RGB')

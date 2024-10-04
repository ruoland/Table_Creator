from logging_config import table_logger, get_memory_handler

from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageChops
import random
import cv2
import numpy as np
from typing import List, Tuple, Optional

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
def apply_noise(image: Image.Image, config) -> Image.Image:
    if config.enable_noise:
        noise_intensity = random.uniform(*config.noise_intensity_range)
        return add_noise(image, noise_intensity)
    return image

def apply_blur(image: Image.Image, config) -> Image.Image:
    if config.enable_blur:
        blur_radius = random.uniform(*config.blur_radius_range)
        return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return image

def apply_brightness(image: Image.Image, config) -> Image.Image:
    if config.enable_brightness_variation:
        brightness_factor = random.uniform(*config.brightness_factor_range)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(brightness_factor)
    return image

def apply_contrast(image: Image.Image, config) -> Image.Image:
    if config.enable_contrast_variation:
        contrast_factor = random.uniform(*config.contrast_factor_range)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(contrast_factor)
    return image

def apply_realistic_effects(image: Image.Image, cells: List[List[float]], table_bbox: List[float], 
                            title_height: int, config) -> Tuple[Image.Image, List[List[float]], List[float], np.ndarray, int, int]:
    """
    이미지에 현실적인 효과를 적용합니다.
    """
    table_logger.debug(f"Applying effects: perspective={config.enable_perspective_transform}, noise={config.enable_noise}, blur={config.enable_blur}")

    transform_matrix = np.eye(3)

    new_width, new_height = image.size

    # 그림자 추가 (원근 변환과 독립적으로)
    if config.enable_shadow:
        all_directions = ['bottom', 'right', 'left', 'top']
        num_directions = random.randint(1, 4)  # 1에서 4 사이의 랜덤한 수의 방향 선택
        shadow_directions = random.sample(all_directions, num_directions)
        shadow_opacity = random.randint(*config.shadow_opacity_range)
        image = add_directional_shadow(image, shadow_directions, config.shadow_blur_radius, 
                                       config.shadow_size_ratio, shadow_opacity)

    image = apply_noise(image, config)
    image = apply_blur(image, config)
    image = apply_brightness(image, config)
    image = apply_contrast(image, config)
    
    return image, cells, table_bbox, transform_matrix, new_width, new_height


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

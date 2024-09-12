import os
import random
from PIL import ImageDraw, ImageFont

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def draw_line(draw, start, end, width, style, color):
    if style == 'double':
        draw.line([start, end], fill=color, width=width)
        offset = max(1, width // 3)
        draw.line([(start[0]+offset, start[1]+offset), (end[0]+offset, end[1]+offset)], fill=color, width=1)
    else:
        draw.line([start, end], fill=color, width=width)

def generate_random_text():
    return ''.join(random.choices('가나다라마바사아자차카타파하', k=random.randint(2, 5)))

def adjust_font_size(draw, text, max_width, max_height, font_path, max_font_size=40):
    font_size = 10
    font = ImageFont.truetype(font_path, size=font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    while text_width < max_width * 0.8 and text_height < max_height * 0.8 and font_size < max_font_size:
        font_size += 1
        font = ImageFont.truetype(font_path, size=font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

    if text_width > max_width * 0.9 or text_height > max_height * 0.9:
        font_size -= 1
        font = ImageFont.truetype(font_path, size=font_size)

    return font, text_width, text_height

import random, string
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from opti_utils import *
from opti_constants import *
from opti_calc import *

def draw_shape(shape_draw, shape_type, size):
    if shape_type == 'rectangle':
        shape_draw.rectangle([0, 0, size-1, size-1], outline=255, width=2)
    elif shape_type == 'ellipse':
        shape_draw.ellipse([0, 0, size-1, size-1], outline=255, width=2)
    elif shape_type == 'triangle':
        shape_draw.polygon([(size//2, 0), (0, size-1), (size-1, size-1)], outline=255, width=2)
    elif shape_type == 'line':
        shape_draw.line([(0, 0), (size-1, size-1)], fill=255, width=2)
    elif shape_type == 'arc':
        shape_draw.arc([0, 0, size-1, size-1], 0, 270, fill=255, width=2)
    elif shape_type == 'polygon':
        points = [(random.randint(0, size-1), random.randint(0, size-1)) for _ in range(5)]
        shape_draw.polygon(points, outline=255, width=2)

def draw_table(draw, cells, table_bbox, bg_color, has_gap, is_imperfect):
    line_widths = []
    border_style = random.choice(BORDER_STYLES)
    
    draw_outer_lines = random.random() > 0.2
    if draw_outer_lines:
        outer_line_thickness = random.randint(1, 5)
        corners = [
            (table_bbox[0], table_bbox[1]),
            (table_bbox[2], table_bbox[1]),
            (table_bbox[2], table_bbox[3]),
            (table_bbox[0], table_bbox[3])
        ]
        for i in range(4):
            start, end = corners[i], corners[(i+1)%4]
            color = get_line_color(bg_color)
            draw_cell_line(draw, start, end, color, outer_line_thickness, is_imperfect, border_style)
            line_widths.append(outer_line_thickness)

    for cell in cells:
        line_thickness = random.randint(1, 3)
        is_merged = len(cell) > 6
        color = get_line_color(bg_color)

        if has_gap or is_merged:
            for start, end in [
                ((cell[0], cell[1]), (cell[2], cell[1])),  # 상단
                ((cell[0], cell[3]), (cell[2], cell[3])),  # 하단
                ((cell[0], cell[1]), (cell[0], cell[3])),  # 좌측
                ((cell[2], cell[1]), (cell[2], cell[3]))   # 우측
            ]:
                draw_cell_line(draw, start, end, color, line_thickness, is_imperfect, border_style)
        else:
            draw_cell_line(draw, (cell[2], cell[1]), (cell[2], cell[3]), color, line_thickness, is_imperfect, border_style)  # 우측
            draw_cell_line(draw, (cell[0], cell[3]), (cell[2], cell[3]), color, line_thickness, is_imperfect, border_style)  # 하단

        line_widths.extend([line_thickness] * 4)

    if is_imperfect and len(cells) > 4:
        num_cells_to_hide = random.randint(1, max(1, len(cells) // 4))
        cells_to_hide = random.sample(cells, num_cells_to_hide)
        for cell in cells_to_hide:
            draw.rectangle([cell[0], cell[1], cell[2], cell[3]], fill=bg_color)

    return line_widths, draw_outer_lines

def draw_cell_line(draw, start, end, color, thickness, is_imperfect, style):
    if style == 'solid':
        draw.line([start, end], fill=color, width=thickness)
    elif style == 'dotted':
        draw_dotted_line(draw, start, end, color, thickness)
    elif style == 'double':
        draw.line([start, end], fill=color, width=thickness)
        offset = thickness + 1
        draw.line([(start[0]+offset, start[1]+offset), (end[0]+offset, end[1]+offset)], fill=color, width=thickness)

    if is_imperfect and random.random() < LINE_BREAK_PROBABILITY:
        break_point = random.uniform(0.3, 0.7)
        mid_x = start[0] + (end[0] - start[0]) * break_point
        mid_y = start[1] + (end[1] - start[1]) * break_point
        draw.line([start, (mid_x, mid_y)], fill=color, width=thickness)
def draw_dotted_line(draw, start, end, color, thickness):
    x1, y1 = start
    x2, y2 = end
    dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    points = int(dist / 5)
    for i in range(0, points, 2):
        x = x1 + (x2 - x1) * i / points
        y = y1 + (y2 - y1) * i / points
        draw.ellipse((x, y, x+thickness, y+thickness), fill=color)

def apply_imperfections(img, cells):
    draw = ImageDraw.Draw(img)
    width, height = img.size

    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.7)))
    
    if random.random() < 0.3:
        for _ in range(random.randint(1, 3)):
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            gap_start = random.uniform(0.3, 0.7)
            gap_end = gap_start + random.uniform(0.1, 0.3)
            
            draw.line((x1, y1, x1 + (x2-x1)*gap_start, y1 + (y2-y1)*gap_start), fill=255, width=1)
            draw.line((x1 + (x2-x1)*gap_end, y1 + (y2-y1)*gap_end, x2, y2), fill=255, width=1)
    
    for cell in cells:
        if random.random() < 0.1:
            x1, y1, x2, y2 = cell[:4]
            start_x, start_y = random.randint(int(x1), int(x2)), random.randint(int(y1), int(y2))
            end_x, end_y = random.randint(int(x1), int(x2)), random.randint(int(y1), int(y2))
            draw.line((start_x, start_y, end_x, end_y), fill=0, width=random.randint(1, 3))
    
    return img


def add_content_to_cells(draw, cells, font_path, bg_color):
    for cell in cells:
        if random.random() < EMPTY_CELL_RATIO:
            continue
        
        content_type = random.choice(['text', 'shapes', 'mixed'])
        
        cell_width = cell[2] - cell[0]
        cell_height = cell[3] - cell[1]
        
        if content_type in ['text', 'mixed']:
            add_text_to_cell(draw, cell, font_path, bg_color)
        if content_type in ['shapes', 'mixed'] and random.random() < SHAPE_GENERATION_RATIO:
            add_shapes_to_cell(draw, cell, bg_color)
            
def adjust_font_size(draw, text, font, max_width, max_height):
    while font.size > 8:  # 최소 폰트 크기 설정
        bbox = draw.multiline_textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width and bbox[3] - bbox[1] <= max_height:
            break
        font = ImageFont.truetype(font.path, font.size - 1)
    return font
def add_text_to_cell(draw, cell, font_path, bg_color):
    cell_width, cell_height = cell[2] - cell[0], cell[3] - cell[1]
    padding = min(cell_width, cell_height) * TEXT_PADDING_RATIO
    max_width = cell_width - 2 * padding
    max_height = cell_height - 2 * padding

    font_size = int(min(cell_width, cell_height) * MAX_FONT_SIZE_RATIO)
    font = ImageFont.truetype(font_path, font_size)
    
    text = random_text()
    text_color = get_line_color(bg_color)
    
    while True:
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] <= max_width and bbox[3] <= max_height:
            break
        if font_size > MIN_FONT_SIZE:
            font_size -= 1
            font = ImageFont.truetype(font_path, font_size)
        else:
            text = text[:int(len(text)*0.9)]  # 텍스트를 10%씩 줄임
            if len(text) == 0:
                text = "."  # 최소 한 글자
            font_size = int(min(cell_width, cell_height) * MAX_FONT_SIZE_RATIO)
            font = ImageFont.truetype(font_path, font_size)

    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = cell[0] + (cell_width - text_width) // 2
    y = cell[1] + (cell_height - text_height) // 2
    
    draw.text((x, y), text, font=font, fill=text_color)
    
    if random.random() < NOISE_PROBABILITY:
        add_noise_around_text(draw, x, y, text_width, text_height, bg_color)

def add_noise_around_text(draw, x, y, width, height, bg_color):
    noise_color = get_line_color(bg_color)
    num_noise = random.randint(MIN_NOISE_COUNT, MAX_NOISE_COUNT)
    for _ in range(num_noise):
        noise_x = x + random.randint(-width//2, width//2)
        noise_y = y + random.randint(-height//2, height//2)
        noise_size = random.randint(1, 3)
        if random.random() < 0.7:
            draw.ellipse([noise_x, noise_y, noise_x+noise_size, noise_y+noise_size], fill=noise_color)
        else:
            end_x = noise_x + random.randint(-5, 5)
            end_y = noise_y + random.randint(-5, 5)
            draw.line([noise_x, noise_y, end_x, end_y], fill=noise_color, width=1)

def add_shapes_to_cell(draw, cell, bg_color):
    cell_width, cell_height = cell[2] - cell[0], cell[3] - cell[1]
    padding = min(cell_width, cell_height) * 0.1
    max_size = min(cell_width, cell_height) - 2 * padding
    color = get_line_color(bg_color)
    
    shape_type = random.choice(['rectangle', 'ellipse', 'triangle', 'line', 'arc', 'polygon'])
    
    shape_img = Image.new('L', (int(max_size), int(max_size)), color=0)
    shape_draw = ImageDraw.Draw(shape_img)
    
    draw_shape(shape_draw, shape_type, int(max_size))
    
    x = cell[0] + (cell_width - max_size) // 2
    y = cell[1] + (cell_height - max_size) // 2
    
    draw.bitmap((int(x), int(y)), shape_img, fill=color)
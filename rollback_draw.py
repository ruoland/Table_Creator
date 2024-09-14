import random, string
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from grrr_utils import *
from grrr_constants import *
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
            draw_cell_line(draw, start, end, color, outer_line_thickness, is_imperfect)
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
                draw_cell_line(draw, start, end, color, line_thickness, is_imperfect)
        else:
            draw_cell_line(draw, (cell[2], cell[1]), (cell[2], cell[3]), color, line_thickness, is_imperfect)  # 우측
            draw_cell_line(draw, (cell[0], cell[3]), (cell[2], cell[3]), color, line_thickness, is_imperfect)  # 하단

        line_widths.extend([line_thickness] * 4)

    if is_imperfect and len(cells) > 4:
        num_cells_to_hide = random.randint(1, max(1, len(cells) // 4))
        cells_to_hide = random.sample(cells, num_cells_to_hide)
        for cell in cells_to_hide:
            draw.rectangle([cell[0], cell[1], cell[2], cell[3]], fill=bg_color)

    return line_widths, draw_outer_lines

def draw_cell_line(draw, start, end, color, thickness, is_imperfect):
    x1, y1 = start
    x2, y2 = end
    
    draw.line((start, end), fill=color, width=thickness)
    
    if is_imperfect:
        if random.random() < 0.2:
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
            protrusion_thickness = random.randint(1, 5)
            protrusion_length = random.randint(1, 3)
            protrusion_color = min(255, color)
            
            if x1 == x2:  # 수직선
                draw.line((mid_x, mid_y, mid_x + protrusion_length, mid_y), fill=protrusion_color, width=protrusion_thickness)
            else:  # 수평선
                draw.line((mid_x, mid_y, mid_x, mid_y + protrusion_length), fill=protrusion_color, width=protrusion_thickness)
        
        if random.random() < 0.2:
            num_dots = random.randint(1, 3)
            for _ in range(num_dots):
                dot_x = random.randint(min(x1, x2), max(x1, x2))
                dot_y = random.randint(min(y1, y2), max(y1, y2))
                dot_size = random.randint(1, 2)
                draw.ellipse((dot_x, dot_y, dot_x + dot_size, dot_y + dot_size), fill=color)

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
            start_x, start_y = random.randint(x1, x2), random.randint(y1, y2)
            end_x, end_y = random.randint(x1, x2), random.randint(y1, y2)
            draw.line((start_x, start_y, end_x, end_y), fill=0, width=random.randint(1, 3))
    
    return img

def add_content_to_cells(draw, cells, font_path, bg_color, empty_cell_ratio=0.2):
    for cell in cells:
        if random.random() < empty_cell_ratio:
            continue
        
        content_type = random.choice(['text', 'shapes', 'mixed'])
        position = random.choice(['center', 'top_left', 'top_right', 'bottom_left', 'bottom_right'])
        
        cell_width = cell[2] - cell[0]
        cell_height = cell[3] - cell[1]
        
        x, y = calculate_position(cell, position)
        
        if content_type in ['text', 'mixed']:
            add_text_to_cell(draw, cell, font_path, bg_color, x, y, position)
        if content_type in ['shapes', 'mixed']:
            add_shapes_to_cell(draw, cell, bg_color, x, y)
            
def adjust_font_size(draw, text, font, max_width, max_height):
    while font.size > 8:  # 최소 폰트 크기 설정
        bbox = draw.multiline_textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width and bbox[3] - bbox[1] <= max_height:
            break
        font = ImageFont.truetype(font.path, font.size - 1)
    return font

def add_text_to_cell(draw, cell, font_path, bg_color, x, y, position):
    cell_width, cell_height = cell[2] - cell[0], cell[3] - cell[1]
    font_size = random.randint(int(cell_height * 0.08), int(cell_height * 0.25))
    font = ImageFont.truetype(font_path, font_size)
    
    text = random_text() if cell_width >= 30 and cell_height >= 30 else random.choice(string.digits + string.punctuation)
    text_color = get_line_color(bg_color)
    
    wrapped_text = wrap_text(text, font, cell_width - 10)
    font = adjust_font_size(draw, wrapped_text, font, cell_width - 10, cell_height - 10)
    
    bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    text_position = calculate_text_position(x, y, text_width, text_height, position)
    
    draw.multiline_text(text_position, wrapped_text, font=font, fill=text_color, align='center')
    add_noise_around_text(draw, text_position[0], text_position[1], text_width, text_height, bg_color)


def add_noise_around_text(draw, x, y, width, height, bg_color):
    noise_color = get_line_color(bg_color)
    num_noise = random.randint(5, 20)
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


def add_shapes_to_cell(draw, cell, bg_color, x, y):
    cell_width, cell_height = cell[2] - cell[0], cell[3] - cell[1]
    max_size = min(cell_width, cell_height) // 2
    color = get_line_color(bg_color)
    
    for _ in range(random.randint(1, 3)):
        shape_type = random.choice(['rectangle', 'ellipse', 'triangle', 'line', 'arc', 'polygon'])
        size = random.randint(max_size // 2, max_size)
        angle = random.randint(0, 360)
        
        shape_img = Image.new('L', (size, size), color=0)
        shape_draw = ImageDraw.Draw(shape_img)
        
        draw_shape(shape_draw, shape_type, size)
        
        rotated_shape = shape_img.rotate(angle, expand=True)
        
        paste_x = max(cell[0], min(x - rotated_shape.width // 2, cell[2] - rotated_shape.width))
        paste_y = max(cell[1], min(y - rotated_shape.height // 2, cell[3] - rotated_shape.height))
        
        draw.bitmap((paste_x, paste_y), rotated_shape, fill=color)

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
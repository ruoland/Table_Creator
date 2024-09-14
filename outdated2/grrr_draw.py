import random, string
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from grrr_utils import *
from grrr_constants import *
from opti_calc import *

def draw_shape(shape_draw: ImageDraw.Draw, shape_type: str, size: int):
    """Draw a shape on the given ImageDraw object."""
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

def draw_table(draw: ImageDraw.Draw, cells: List[List[int]], table_bbox: List[int], 
               bg_color: int, has_gap: bool, is_imperfect: bool) -> Tuple[List[int], bool]:
    """Draw table lines and apply imperfections if specified."""
    line_widths = []
    draw_outer_lines = random.random() > OUTER_LINE_PROBABILITY

    if draw_outer_lines:
        outer_line_thickness = random.randint(MIN_LINE_WIDTH, MAX_LINE_WIDTH)
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
        line_thickness = random.randint(MIN_LINE_WIDTH, MAX_LINE_WIDTH)
        is_merged = len(cell) > 6
        color = get_line_color(bg_color)

        if has_gap or is_merged:
            for start, end in [
                ((cell[0], cell[1]), (cell[2], cell[1])),  # top
                ((cell[0], cell[3]), (cell[2], cell[3])),  # bottom
                ((cell[0], cell[1]), (cell[0], cell[3])),  # left
                ((cell[2], cell[1]), (cell[2], cell[3]))   # right
            ]:
                draw_cell_line(draw, start, end, color, line_thickness, is_imperfect)
        else:
            draw_cell_line(draw, (cell[2], cell[1]), (cell[2], cell[3]), color, line_thickness, is_imperfect)  # right
            draw_cell_line(draw, (cell[0], cell[3]), (cell[2], cell[3]), color, line_thickness, is_imperfect)  # bottom

        line_widths.extend([line_thickness] * 4)

    if is_imperfect and len(cells) > 4:
        apply_imperfections(draw, cells)

    return line_widths, draw_outer_lines
def draw_cell_line(draw: ImageDraw.Draw, start: Tuple[int, int], end: Tuple[int, int], 
                   color: int, thickness: int, is_imperfect: bool):
    """Draw a cell line with optional imperfections."""
    x1, y1 = start
    x2, y2 = end
    
    draw.line((start, end), fill=color, width=thickness)
    
    if is_imperfect:
        apply_line_imperfections(draw, x1, y1, x2, y2, color, thickness)

def apply_line_imperfections(draw: ImageDraw.Draw, x1: int, y1: int, x2: int, y2: int, 
                             color: int, thickness: int):
    """Apply imperfections to a line."""
    if random.random() < PROTRUSION_PROBABILITY:
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
        protrusion_thickness = random.randint(1, 5)
        protrusion_length = random.randint(1, 3)
        protrusion_color = min(255, color)
        
        if x1 == x2:  # vertical line
            draw.line((mid_x, mid_y, mid_x + protrusion_length, mid_y), fill=protrusion_color, width=protrusion_thickness)
        else:  # horizontal line
            draw.line((mid_x, mid_y, mid_x, mid_y + protrusion_length), fill=protrusion_color, width=protrusion_thickness)
    
    if random.random() < DOT_PROBABILITY:
        num_dots = random.randint(1, 3)
        for _ in range(num_dots):
            dot_x = random.randint(min(x1, x2), max(x1, x2))
            dot_y = random.randint(min(y1, y2), max(y1, y2))
            dot_size = random.randint(1, 2)
            draw.ellipse((dot_x, dot_y, dot_x + dot_size, dot_y + dot_size), fill=color)


def apply_imperfections(img: Image.Image, cells: List[List[int]]) -> Image.Image:
    """Apply various imperfections to the image."""
    draw = ImageDraw.Draw(img)
    width, height = img.size

    if random.random() < BLUR_PROBABILITY:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(MIN_BLUR, MAX_BLUR)))
    
    if random.random() < LINE_IMPERFECTION_PROBABILITY:
        add_imperfect_lines(draw, width, height)
    
    add_cell_imperfections(draw, cells)
    
    return img

def add_imperfect_lines(draw: ImageDraw.Draw, width: int, height: int):
    """Add imperfect lines to the image."""
    for _ in range(random.randint(1, MAX_IMPERFECT_LINES)):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        gap_start = random.uniform(0.3, 0.7)
        gap_end = gap_start + random.uniform(0.1, 0.3)
        
        draw.line((x1, y1, x1 + (x2-x1)*gap_start, y1 + (y2-y1)*gap_start), fill=255, width=1)
        draw.line((x1 + (x2-x1)*gap_end, y1 + (y2-y1)*gap_end, x2, y2), fill=255, width=1)

def add_cell_imperfections(draw: ImageDraw.Draw, cells: List[List[int]]):
    """Add imperfections to individual cells."""
    for cell in cells:
        if random.random() < CELL_IMPERFECTION_PROBABILITY:
            x1, y1, x2, y2 = cell[:4]
            start_x, start_y = random.randint(x1, x2), random.randint(y1, y2)
            end_x, end_y = random.randint(x1, x2), random.randint(y1, y2)
            draw.line((start_x, start_y, end_x, end_y), fill=0, width=random.randint(1, 3))

def add_content_to_cells(draw: ImageDraw.Draw, cells: List[List[int]], font_path: str, bg_color: int):
    """Add content (text or shapes) to cells."""
    for cell in cells:
        if random.random() < EMPTY_CELL_RATIO:
            continue
        
        content_type = random.choice(['text', 'shapes', 'mixed'])
        position = random.choice(['center', 'top_left', 'top_right', 'bottom_left', 'bottom_right'])
        
        x, y = calculate_position(cell, position)
        
        if content_type in ['text', 'mixed']:
            add_text_to_cell(draw, cell, font_path, bg_color, x, y, position)
        if content_type in ['shapes', 'mixed']:
            add_shapes_to_cell(draw, cell, bg_color, x, y)

def add_text_to_cell(draw: ImageDraw.Draw, cell: List[int], font_path: str, bg_color: int, x: int, y: int, position: str):
    """Add text to a cell."""
    cell_width, cell_height = cell[2] - cell[0], cell[3] - cell[1]
    font_size = random.randint(int(cell_height * MIN_FONT_SIZE_RATIO), int(cell_height * MAX_FONT_SIZE_RATIO))
    font = ImageFont.truetype(font_path, font_size)
    
    text = random_text() if cell_width >= MIN_CELL_SIZE_FOR_TEXT and cell_height >= MIN_CELL_SIZE_FOR_TEXT else random.choice(string.digits + string.punctuation)
    text_color = get_line_color(bg_color)
    
    wrapped_text = wrap_text(text, font, cell_width - TEXT_PADDING)
    font = adjust_font_size(draw, wrapped_text, font, cell_width - TEXT_PADDING, cell_height - TEXT_PADDING)
    
    bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    text_position = calculate_text_position(x, y, text_width, text_height, position)
    
    draw.multiline_text(text_position, wrapped_text, font=font, fill=text_color, align='center')
    add_noise_around_text(draw, text_position[0], text_position[1], text_width, text_height, bg_color)

def adjust_font_size(draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont, max_width: int, max_height: int) -> ImageFont.FreeTypeFont:
    """Adjust font size to fit within given dimensions."""
    while font.size > MIN_FONT_SIZE:
        bbox = draw.multiline_textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width and bbox[3] - bbox[1] <= max_height:
            break
        font = ImageFont.truetype(font.path, font.size - 1)
    return font

def add_noise_around_text(draw: ImageDraw.Draw, x: int, y: int, width: int, height: int, bg_color: int):
    """Add noise around text."""
    noise_color = get_line_color(bg_color)
    num_noise = random.randint(MIN_NOISE, MAX_NOISE)
    for _ in range(num_noise):
        noise_x = x + random.randint(-width//2, width//2)
        noise_y = y + random.randint(-height//2, height//2)
        noise_size = random.randint(MIN_NOISE_SIZE, MAX_NOISE_SIZE)
        if random.random() < DOT_NOISE_PROBABILITY:
            draw.ellipse([noise_x, noise_y, noise_x+noise_size, noise_y+noise_size], fill=noise_color)
        else:
            end_x = noise_x + random.randint(-MAX_LINE_NOISE_LENGTH, MAX_LINE_NOISE_LENGTH)
            end_y = noise_y + random.randint(-MAX_LINE_NOISE_LENGTH, MAX_LINE_NOISE_LENGTH)
            draw.line([noise_x, noise_y, end_x, end_y], fill=noise_color, width=1)

def add_shapes_to_cell(draw: ImageDraw.Draw, cell: List[int], bg_color: int, x: int, y: int):
    """Add shapes to a cell."""
    cell_width, cell_height = cell[2] - cell[0], cell[3] - cell[1]
    max_size = min(cell_width, cell_height) // 2
    color = get_line_color(bg_color)
    
    for _ in range(random.randint(MIN_SHAPES_PER_CELL, MAX_SHAPES_PER_CELL)):
        shape_type = random.choice(SHAPE_TYPES)
        size = random.randint(max_size // 2, max_size)
        angle = random.randint(0, 360)
        
        shape_img = Image.new('L', (size, size), color=0)
        shape_draw = ImageDraw.Draw(shape_img)
        
        draw_shape(shape_draw, shape_type, size)
        
        rotated_shape = shape_img.rotate(angle, expand=True)
        
        paste_x = max(cell[0], min(x - rotated_shape.width // 2, cell[2] - rotated_shape.width))
        paste_y = max(cell[1], min(y - rotated_shape.height // 2, cell[3] - rotated_shape.height))
        
        draw.bitmap((paste_x, paste_y), rotated_shape, fill=color)

def draw_shape(shape_draw: ImageDraw.Draw, shape_type: str, size: int):
    """Draw a specific shape."""
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

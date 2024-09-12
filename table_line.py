import random
from PIL import Image, ImageDraw

def generate_line_thickness():
    return random.uniform(0.5, 3.0)

def generate_line_style():
    styles = ['solid', 'dashed', 'dotted']
    return random.choice(styles)

def draw_line(draw, start, end, color, thickness, style):
    if style == 'solid':
        draw.line([start, end], fill=color, width=int(thickness))
    elif style == 'dashed':
        draw_dashed_line(draw, start, end, color, thickness)
    elif style == 'dotted':
        draw_dotted_line(draw, start, end, color, thickness)

def draw_dashed_line(draw, start, end, color, thickness):
    x1, y1 = start
    x2, y2 = end
    dash_length = int(thickness * 3)
    gap_length = int(thickness * 2)
    
    if x1 == x2:  # vertical line
        steps = abs(y2 - y1)
        direction = 1 if y2 > y1 else -1
        for i in range(0, steps, dash_length + gap_length):
            y_start = y1 + i * direction
            y_end = min(y_start + dash_length * direction, y2) if direction > 0 else max(y_start + dash_length * direction, y2)
            draw.line([(x1, y_start), (x1, y_end)], fill=color, width=int(thickness))
    else:  # horizontal line
        steps = abs(x2 - x1)
        direction = 1 if x2 > x1 else -1
        for i in range(0, steps, dash_length + gap_length):
            x_start = x1 + i * direction
            x_end = min(x_start + dash_length * direction, x2) if direction > 0 else max(x_start + dash_length * direction, x2)
            draw.line([(x_start, y1), (x_end, y1)], fill=color, width=int(thickness))

def draw_dotted_line(draw, start, end, color, thickness):
    x1, y1 = start
    x2, y2 = end
    dot_gap = int(thickness * 2)
    
    if x1 == x2:  # vertical line
        steps = abs(y2 - y1)
        direction = 1 if y2 > y1 else -1
        for i in range(0, steps, dot_gap):
            y = y1 + i * direction
            draw.ellipse([(x1 - thickness/2, y - thickness/2), (x1 + thickness/2, y + thickness/2)], fill=color)
    else:  # horizontal line
        steps = abs(x2 - x1)
        direction = 1 if x2 > x1 else -1
        for i in range(0, steps, dot_gap):
            x = x1 + i * direction
            draw.ellipse([(x - thickness/2, y1 - thickness/2), (x + thickness/2, y1 + thickness/2)], fill=color)

def draw_cell_border(draw, cell, line_color):
    x1, y1, x2, y2 = cell
    for side in ['top', 'right', 'bottom', 'left']:
        thickness = generate_line_thickness()
        style = generate_line_style()
        if side == 'top':
            draw_line(draw, (x1, y1), (x2, y1), line_color, thickness, style)
        elif side == 'right':
            draw_line(draw, (x2, y1), (x2, y2), line_color, thickness, style)
        elif side == 'bottom':
            draw_line(draw, (x1, y2), (x2, y2), line_color, thickness, style)
        elif side == 'left':
            draw_line(draw, (x1, y1), (x1, y2), line_color, thickness, style)

# 사용 예시
def create_table_with_varied_lines(width, height, rows, cols, bg_color, line_color):
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    cell_width = width // cols
    cell_height = height // rows

    for row in range(rows):
        for col in range(cols):
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            draw_cell_border(draw, (x1, y1, x2, y2), line_color)

    return img

# 테스트
table_img = create_table_with_varied_lines(800, 600, 5, 7, (255, 255, 255), (0, 0, 0))
table_img.save('varied_line_table.png')

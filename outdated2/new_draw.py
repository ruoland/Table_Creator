import math
def draw_styled_line(draw, start, end, style, width, fill):
    if style == 'solid':
        draw.line([start, end], width=width, fill=fill)
    elif style == 'dashed':
        dash_length = 10
        space_length = 5
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx**2 + dy**2)
        unit_x = dx / length
        unit_y = dy / length
        
        current_length = 0
        while current_length < length:
            dash_end = min(current_length + dash_length, length)
            dash_x1 = x1 + current_length * unit_x
            dash_y1 = y1 + current_length * unit_y
            dash_x2 = x1 + dash_end * unit_x
            dash_y2 = y1 + dash_end * unit_y
            draw.line([(dash_x1, dash_y1), (dash_x2, dash_y2)], width=width, fill=fill)
            current_length = dash_end + space_length
    elif style == 'dotted':
        dot_spacing = 5
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx**2 + dy**2)
        unit_x = dx / length
        unit_y = dy / length
        
        for i in range(0, int(length), dot_spacing):
            x = x1 + i * unit_x
            y = y1 + i * unit_y
            draw.ellipse([(x-width/2, y-width/2), (x+width/2, y+width/2)], fill=fill)

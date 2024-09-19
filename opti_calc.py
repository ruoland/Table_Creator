


def calculate_position(cell, position):
    cell_width = cell[2] - cell[0]
    cell_height = cell[3] - cell[1]
    
    if position == 'center':
        return cell[0] + cell_width // 2, cell[1] + cell_height // 2
    elif position == 'top_left':
        return cell[0] + int(cell_width * 0.1), cell[1] + int(cell_height * 0.1)
    elif position == 'top_right':
        return cell[2] - int(cell_width * 0.1), cell[1] + int(cell_height * 0.1)
    elif position == 'bottom_left':
        return cell[0] + int(cell_width * 0.1), cell[3] - int(cell_height * 0.1)
    else:  # bottom_right
        return cell[2] - int(cell_width * 0.1), cell[3] - int(cell_height * 0.1)
    
def calculate_text_position(cell_x, cell_y, cell_width, cell_height, text_width, text_height, position):
    if position == 'center':
        return (cell_x + (cell_width - text_width) // 2, cell_y + (cell_height - text_height) // 2)
    elif position == 'top_left':
        return (cell_x, cell_y)
    elif position == 'top_right':
        return (cell_x + cell_width - text_width, cell_y)
    elif position == 'bottom_left':
        return (cell_x, cell_y + cell_height - text_height)
    else:  # bottom_right
        return (cell_x + cell_width - text_width, cell_y + cell_height - text_height)

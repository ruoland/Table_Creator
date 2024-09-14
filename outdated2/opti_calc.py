


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
    

def calculate_text_position(x, y, text_width, text_height, position):
    if position == 'center':
        return (x - text_width // 2, y - text_height // 2)
    elif position == 'top_left':
        return (x, y)
    elif position == 'top_right':
        return (x - text_width, y)
    elif position == 'bottom_left':
        return (x, y - text_height)
    else:  # bottom_right
        return (x - text_width, y - text_height)

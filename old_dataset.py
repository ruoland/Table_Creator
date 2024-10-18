
def get_perspective_points(width: int, height: int, intensity: float, direction: Optional[str] = None) -> np.ndarray:
    """
    원근 변환을 위한 목표 포인트를 생성합니다.
    """
    max_offset_x = width * intensity * 0.15
    max_offset_y = height * intensity * 0.15
    
    if direction is None:
        top_left_x = random.uniform(0, max_offset_x)
        top_left_y = random.uniform(0, max_offset_y)
        top_right_x = random.uniform(width - max_offset_x, width)
        top_right_y = random.uniform(0, max_offset_y)
        bottom_right_x = random.uniform(width - max_offset_x, width)
        bottom_right_y = random.uniform(height - max_offset_y, height)
        bottom_left_x = random.uniform(0, max_offset_x)
        bottom_left_y = random.uniform(height - max_offset_y, height)
    else:
        # 지정된 방향에 따른 변형
        if direction in ['left', 'top_left', 'bottom_left']:
            top_left_x = bottom_left_x = random.uniform(max_offset_x/2, max_offset_x)
        else:
            top_left_x = bottom_left_x = 0

        if direction in ['right', 'top_right', 'bottom_right']:
            top_right_x = bottom_right_x = random.uniform(width - max_offset_x, width - max_offset_x/2)
        else:
            top_right_x = bottom_right_x = width

        if direction in ['top', 'top_left', 'top_right']:
            top_left_y = top_right_y = random.uniform(max_offset_y/2, max_offset_y)
        else:
            top_left_y = top_right_y = 0

        if direction in ['bottom', 'bottom_left', 'bottom_right']:
            bottom_left_y = bottom_right_y = random.uniform(height - max_offset_y, height - max_offset_y/2)
        else:
            bottom_left_y = bottom_right_y = height

    return np.float32([
        [top_left_x, top_left_y],
        [top_right_x, top_right_y],
        [bottom_right_x, bottom_right_y],
        [bottom_left_x, bottom_left_y]
    ])
if config.enable_perspective_transform:
        try:
            image, cells, table_bbox, transform_matrix = apply_perspective_transform(
                image, cells, table_bbox, title_height, config.perspective_intensity, config
            )
            logging.info("Perspective transform applied successfully")
        except Exception as e:
            logging.error(f"Perspective transform failed: {e}")
            

def get_shadow_direction(dst_points: np.ndarray) -> List[str]:
    """
    변환된 포인트를 기반으로 그림자 방향을 결정합니다.
    """
    top_left, top_right, bottom_right, bottom_left = dst_points
    
    directions = []
    if top_left[1] > bottom_left[1]:
        directions.append('top')
    if top_right[1] > top_left[1]:
        directions.append('right')
    if bottom_right[0] < top_right[0]:
        directions.append('left')
    if bottom_right[1] > top_right[1]:
        directions.append('bottom')
    
    return directions if directions else ['bottom']  # 기본값으로 'bottom' 반환

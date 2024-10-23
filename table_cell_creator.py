import random
from dataset_utils import is_overlapping
from logging_config import table_logger


def log_cell_coordinates(cells, stage):
    table_logger.debug(f"=== Cell coordinates at {stage} ===")
    for cell in cells:
        table_logger.debug(f"Cell: row={cell['row']}, col={cell['col']}, x1={cell['x1']}, y1={cell['y1']}, x2={cell['x2']}, y2={cell['y2']}")
    table_logger.debug("=" * 50)
    
def can_merge(config):
    if config.enable_cell_merging:
        if config.enable_horizontal_merge or config.enable_vertical_merge:
            return True
        else:
            config.enable_vertical_merge = True
            return True
    return False


def create_merged_cell(cells, start_row, start_col, merge_rows, merge_cols, cols, config):
    # total_rows = len(cells) // cols
    
    # # 병합 범위가 테이블을 벗어나지 않는지 확인
    # if (start_row + merge_rows > total_rows) or (start_col + merge_cols > cols):
    #     return None  # 병합 범위가 테이블을 벗어나면 None 반환
    
    base_cell_index = start_row * cols + start_col
    end_cell_index = (start_row + merge_rows - 1) * cols + (start_col + merge_cols - 1)
    
    # 인덱스가 유효한지 확인
    if base_cell_index >= len(cells) or end_cell_index >= len(cells):
        return None  # 유효하지 않은 인덱스면 None 반환
    
    base_cell = cells[base_cell_index]
    end_cell = cells[end_cell_index]
    
    new_cell_info = {
        'x1': base_cell['x1'],
        'y1': base_cell['y1'],
        'x2': end_cell['x2'],
        'y2': end_cell['y2'],
        'row': start_row,
        'col': start_col,
        'is_merged': True,
        'is_header': base_cell['is_header'],
        'original_height': end_cell['y2'] - base_cell['y1'],
        'overflow': None,
        'merge_rows': merge_rows,
        'merge_cols': merge_cols,
        'cell_type': 'merged_cell',
    }
    
    # 병합된 셀의 높이와 너비 확인
    if (new_cell_info['x2'] - new_cell_info['x1'] < config.min_cell_width or 
        new_cell_info['y2'] - new_cell_info['y1'] < config.min_cell_height):
        return None  # 최소 크기를 만족하지 않으면 None 반환
    
    return new_cell_info
def perform_cell_merging(cells, rows, cols, config):
    merged_cells = cells.copy()
    merged_areas = []
    header_cells = []  # 헤더 셀들을 저장할 리스트
    log_cell_coordinates(merged_cells, "Start of merge_cells")
    
    # 헤더 셀 식별 및 저장
    for cell in merged_cells:
        if ((cell['row'] == 0 and config.table_type in ['header_row', 'header_both']) or
            (cell['col'] == 0 and config.table_type in ['header_column', 'header_both'])):
            header_cells.append(cell)

    if config.enable_horizontal_merge:
        for row in range(rows):
            if row == 0 and config.table_type in ['header_row', 'header_both']:
                continue  # 헤더 행 스킵
            col = 0 
            while col < cols - 3:
                if col == 0 and config.table_type in ['header_column', 'header_both']:
                    col += 1  # 헤더 열 스킵
                    continue
                if random.random() < config.horizontal_merge_probability:
                    merge_cols = random.randint(2, min(config.max_horizontal_merge, cols - col))
                    merge_area = (row, col, row + 3, col + merge_cols) 
                    if not is_overlapping_with_merged_or_overflow(merge_area, merged_areas, merged_cells):
                        new_cell_info = create_merged_cell(merged_cells, row, col, 3, merge_cols, cols, config) 
                        if new_cell_info:
                            merged_cells[row * cols + col] = new_cell_info 
                            merged_areas.append(merge_area) 
                            col += merge_cols 
                            continue 
                col += 3 

    if config.enable_vertical_merge:
        for col in range(cols):
            if col == 0 and config.table_type in ['header_column', 'header_both']:
                continue  # 헤더 열 스킵
            row = 0
            while row < rows - 2:
                if row == 0 and config.table_type in ['header_row', 'header_both']:
                    row += 1  # 헤더 행 스킵
                    continue
                if random.random() < config.vertical_merge_probability:
                    max_merge = min(config.max_vertical_merge, rows - row)
                    if max_merge < 2:
                        break
                    merge_rows = random.randint(2, max_merge)
                    merge_area = (row, col, row + merge_rows, col + 1)
                    
                    # 병합된 셀과의 겹침 확인
                    new_cell_info = create_merged_cell(merged_cells, row, col, merge_rows, 1, cols, config)
                    if new_cell_info and not is_overlapping_with_merged_cells(new_cell_info, merged_cells):
                        merged_cells[row * cols + col] = new_cell_info
                        merged_areas.append(merge_area)

                        row += merge_rows
                        continue
                row += 1

    # 병합된 셀 정보 업데이트 및 중복 셀 제거
    for area in merged_areas:
        row_start, col_start, row_end, col_end = area
        base_cell = merged_cells[row_start * cols + col_start]
        base_cell['cell_type'] = 'merged_cell'
        base_cell['is_merged'] = True
        base_cell['merge_rows'] = row_end - row_start
        base_cell['merge_cols'] = col_end - col_start

        # 병합된 영역의 다른 셀들 제거
        for r in range(row_start, row_end):
            for c in range(col_start, col_end):
                if r != row_start or c != col_start:
                    merged_cells[r * cols + c] = None

    # None이 아닌 셀과 헤더 셀 모두 유지
    merged_cells = [cell for cell in merged_cells if cell is not None]
    
    return merged_cells


def is_overlapping_with_merged_or_overflow(new_area, merged_areas, cells):
    row_start, col_start, row_end, col_end = new_area
    
    # 기존 병합 영역과의 겹침 확인
    if any(is_overlapping(new_area, area) for area in merged_areas):
        return True
    
    # 오버플로우된 셀과의 겹침 확인
    for cell in cells:
        if cell and (cell.get('overflow') or (cell.get('overflow') and cell.get('is_merged'))):
            cell_row, cell_col = cell['row'], cell['col']
            cell_rows = cell.get('merge_rows', 1)
            cell_cols = cell.get('merge_cols', 1)
            
            overflow_y1 = cell.get('overflow_y1', cell['y1'])
            overflow_y2 = cell.get('overflow_y2', cell['y2'])
            
            # 수평 겹침 확인
            horizontal_overlap = col_start < cell_col + cell_cols and col_end > cell_col
            
            # 수직 겹침 확인 (기본 영역 + 오버플로우 영역)
            vertical_overlap = (
                (row_start < cell_row + cell_rows and row_end > cell_row) or
                (row_start < overflow_y2 and row_end > overflow_y1)
            )
            
            if horizontal_overlap and vertical_overlap:
                return True
    
    return False

def is_overlapping_with_nearby_cells(current_cell, all_cells, overflow_direction, height_up, height_down, config):
    current_row, current_col = current_cell['row'], current_cell['col']
    merge_rows = current_cell.get('merge_rows', 1)
    merge_cols = current_cell.get('merge_cols', 1)

    # 오버플로우 영역 계산
    overflow_y1 = current_cell['y1'] - height_up if overflow_direction in ['up', 'both'] else current_cell['y1']
    overflow_y2 = current_cell['y2'] + height_down if overflow_direction in ['down', 'both'] else current_cell['y2']

    # 검사할 행 범위 계산
    start_row = max(0, current_row - 1) if overflow_direction in ['up', 'both'] else current_row
    end_row = min(current_row + merge_rows + 1, config.total_rows) if overflow_direction in ['down', 'both'] else current_row + merge_rows

    for cell in all_cells:
        if cell == current_cell:
            continue

        cell_row, cell_col = cell['row'], cell['col']
        cell_merge_rows = cell.get('merge_rows', 1)
        cell_merge_cols = cell.get('merge_cols', 1)

        # 근처 셀만 검사 (병합된 셀의 전체 영역 고려)
        if not (start_row <= cell_row + cell_merge_rows - 1 and 
                cell_row < end_row and
                current_col - 1 <= cell_col + cell_merge_cols - 1 and 
                cell_col < current_col + merge_cols + 1):
            continue

        # 수평 겹침 확인 (병합된 셀의 전체 너비 고려)
        horizontal_overlap = (current_cell['x1'] < cell['x2'] + (cell_merge_cols - 1) * (cell['x2'] - cell['x1']) and 
                              current_cell['x2'] > cell['x1'])

        # 수직 겹침 확인 (병합된 셀의 전체 높이와 오버플로우 고려)
        cell_y2 = cell['y2'] + (cell_merge_rows - 1) * (cell['y2'] - cell['y1'])
        
        # 셀의 오버플로우 영역 계산
        cell_overflow_y1 = cell.get('overflow_y1', cell['y1']) if cell.get('overflow') else cell['y1']
        cell_overflow_y2 = cell.get('overflow_y2', cell_y2) if cell.get('overflow') else cell_y2

        vertical_overlap = (overflow_y1 < cell_overflow_y2 and overflow_y2 > cell_overflow_y1)

        # 겹침 조건 확인
        if horizontal_overlap and vertical_overlap:
            # 다른 오버플로우된 셀과의 겹침 확인 (병합된 오버플로우 셀 포함)
            if cell.get('overflow') or (cell.get('is_merged', False) and cell.get('overflow')):
                return True
            
            # 병합된 셀과의 겹침 확인
            if cell.get('is_merged', False):
                return True
            
            # 헤더 셀과의 겹침 확인
            if cell.get('is_header', False):
                return True

    return False


#병합 셀 처리 후, 아래 함수 호출
def plan_cell_overflow(cells, config):
    if not config.enable_overflow:
        return cells

    overflow_count = 0
    affected_cells = set()
    cell_dict = {(cell['row'], cell['col']): cell for cell in cells}

    # 셀을 무작위 순서로 처리하되, 병합된 셀을 우선적으로 처리
    shuffled_cells = sorted(cells, key=lambda x: x.get('is_merged', False), reverse=True)
    shuffled_cells = random.sample(shuffled_cells, len(shuffled_cells))

    
    for cell in shuffled_cells:
        #헤더 셀이 아니고, 행과 열을 초과하지 않고, 영향 받는 셀이 아닌 경우
        if (not cell['is_header'] and 
            0 <= cell['row'] < config.total_rows and  
            0 <= cell['col'] < config.total_cols and  
            (cell['row'], cell['col']) not in affected_cells):
            
            if cell['cell_type'] == 'overflow_cell' and cell.get('is_merged', False):
                continue
            #확률 계산하기, 만약 병합된 셀인 경우 전용 확률을 사용, 아닌 경우 일반 오버플로 확률 사용
            overflow_prob = config.overflow_probability

            if overflow_count > config.total_rows / 3 or overflow_count > config.total_cols / 3:
                continue
            
                
            if random.random() < overflow_prob:
                table_logger.warning(f"셀 처리 중: is_merged={cell['is_merged']}")

                #병합된 행과 열 길이를 가져옴, 없으면 1
                merge_rows = cell.get('merge_rows', 1)
                merge_cols = cell.get('merge_cols', 1)
                
                #셀이 0 행에 있다면 가능한 건 아래 방향
                #셀이 마지막 행에 있다면 위쪽으로만 가능
                #셀이 중간 행에 있다면 상하좌우 가능
                possible_directions = ['down'] if cell['row'] == 0 else ['up'] if cell['row'] + merge_rows == config.total_rows else ['up', 'down', 'both']
                
                #가능한 방향 중 아무거나 골라서 선택함
                direction = random.choice(possible_directions)
                
                #높이
                min_height = min(config.min_overflow_height, config.max_cell_height)
                max_height = max(config.min_overflow_height, config.max_cell_height)
                overflow_height = random.randint(min_height, max_height)

                height_up = overflow_height if direction == 'up' else (overflow_height // 2 if direction == 'both' else 0)
                height_down = overflow_height if direction == 'down' else (overflow_height - height_up if direction == 'both' else 0)
                
                #table_logger.warning(f"오버플로우 계획: is_merged={cell['is_merged']}direction={direction}, height_up={height_up}, height_down={height_down}")
                if cell['is_merged'] ==  True:
                   table_logger.warning(f'오버플로우 병합된 셀 생성 시도 결과{not is_overlapping_with_nearby_cells(cell, cells, direction, height_up, height_down, config)}') 

                #주변 셀과 방향을 고려하여 겹치는 게 없다면
                if not is_overlapping_with_nearby_cells(cell, cells, direction, height_up, height_down, config):
                    cell['overflow'] = {
                        'direction': direction,
                        'height_up': height_up,
                        'height_down': height_down,
                    }
                    cell['overflow_y1'] = cell['y1'] - height_up if direction in ['up', 'both'] else cell['y1']
                    cell['overflow_y2'] = cell['y2'] + height_down if direction in ['down', 'both'] else cell['y2']
                    cell['overflow_applied'] = True

                    if cell['is_merged']:
                        cell['cell_type'] = 'overflow_cell'
                        table_logger.warning(f"병합 오버플로우 계획: is_merged={cell['is_merged']}direction={direction}, height_up={height_up}, height_down={height_down}")
                        
                    else:
                        cell['cell_type'] = 'overflow_cell'
                        overflow_count += 1

                    for r in range(cell['row'], cell['row'] + merge_rows):
                        for c in range(cell['col'], cell['col'] + merge_cols):
                            affected_cells.add((r, c))
                else:
                    table_logger.debug(f"오버플로우 적용 불가: row={cell['row']}, col={cell['col']}")


    table_logger.warning(f"오버플로우 계획 완료: 총 {overflow_count}개 일반 셀에 적용")
    return cells

def is_overlapping_with_merged_cells(new_cell, all_cells):
    new_row_start = new_cell['row']
    new_col_start = new_cell['col']
    new_row_end = new_row_start + new_cell.get('merge_rows', 1)
    new_col_end = new_col_start + new_cell.get('merge_cols', 1)
    
    new_y1 = new_cell.get('overflow_y1', new_cell['y1'])
    new_y2 = new_cell.get('overflow_y2', new_cell['y2'])
    
    for cell in all_cells:
        if cell == new_cell:
            continue
        
        if cell.get('is_merged', False) or cell.get('overflow'):
            cell_row_start = cell['row']
            cell_col_start = cell['col']
            cell_row_end = cell_row_start + cell.get('merge_rows', 1)
            cell_col_end = cell_col_start + cell.get('merge_cols', 1)
            
            cell_y1 = cell.get('overflow_y1', cell['y1'])
            cell_y2 = cell.get('overflow_y2', cell['y2'])

            # 겹침 조건 확인 (열, 행, 그리고 실제 y 좌표 고려)
            if (new_col_start < cell_col_end and new_col_end > cell_col_start and
                new_row_start < cell_row_end and new_row_end > cell_row_start and
                new_y1 < cell_y2 and new_y2 > cell_y1):
                return True
    return False
def is_overlapping_with_other_overflows(cell, all_cells, direction, height_up, height_down):
    overflow_y1 = cell['y1'] - height_up if direction in ['up', 'both'] else cell['y1']
    overflow_y2 = cell['y2'] + height_down if direction in ['down', 'both'] else cell['y2']
    
    for other_cell in all_cells:
        if other_cell == cell:
            continue
        if 'overflow' in other_cell or other_cell.get('is_merged', False):
            other_overflow_y1 = other_cell.get('overflow_y1', other_cell['y1'])
            other_overflow_y2 = other_cell.get('overflow_y2', other_cell['y2'])
            other_x2 = other_cell['x2'] + (other_cell.get('merge_cols', 1) - 1) * (other_cell['x2'] - other_cell['x1'])
            
            if (overflow_y1 < other_overflow_y2 and overflow_y2 > other_overflow_y1) and \
               (cell['x1'] < other_x2 and cell['x2'] > other_cell['x1']):
                return True
    return False


def check_overflow_validity(cell, all_cells, direction, height_up, height_down, config, table_bbox):
    return (not is_overlapping_with_nearby_cells(cell, all_cells, direction, height_up, height_down, config) and
            not is_overlapping_with_merged_cells(cell, all_cells) and
            is_within_table_bounds(cell, table_bbox, direction, height_up, height_down) and
            not is_overlapping_with_other_overflows(cell, all_cells, direction, height_up, height_down))

def is_valid_overflow_candidate(cell, all_cells, config, table_bbox):
    directions = ['up', 'down'] if cell['row'] > 0 and cell['row'] < config.total_rows - 1 else ['down'] if cell['row'] == 0 else ['up']
    cell_height = cell['y2'] - cell['y1']
    
    for direction in directions:
        try:
            overflow_height = get_random_overflow_height(cell_height, config)
        except ValueError:
            continue  # 유효한 오버플로우 높이를 얻을 수 없는 경우 다음 방향으로 넘어갑니다.
        
        height_up = overflow_height if direction == 'up' else 0
        height_down = overflow_height if direction == 'down' else 0
        
        if check_overflow_validity(cell, all_cells, direction, height_up, height_down, config, table_bbox):
            return True
    return False

def is_within_table_bounds(cell, table_bbox, direction, height_up, height_down):
    table_y1, table_y2 = table_bbox[1], table_bbox[3]
    
    if direction == 'up' or direction == 'both':
        if cell['y1'] - height_up < table_y1:
            return False
    
    if direction == 'down' or direction == 'both':
        if cell['y2'] + height_down > table_y2:
            return False
    
    return True
def merge_cells(cells, rows, cols, config, table_bbox):
    if rows <= 3 or cols <= 3 or not can_merge(config):
        return cells
    
    merged_cells = perform_cell_merging(cells, rows, cols, config)
    return merged_cells


def get_random_overflow_height(cell_height, config):
    min_height = max(config.min_overflow_height, config.max_cell_height + 1)
    max_height = min(cell_height - random.randint(2, 4), config.max_overflow_height)
    
    if min_height > max_height:
        return max_height  # 또는 min_height, 상황에 따라 선택
    
    return random.randint(min_height, max_height)
def apply_overflow_to_cells(candidates, cells, config, table_bbox):
    # 병합된 셀 먼저 처리
    merged_candidates = [cell for cell in candidates if cell.get('is_merged', False)]
    for cell in merged_candidates:
        apply_overflow_to_merged_cell(cell, cells, config, table_bbox)
    
    # 일반 셀 처리
    normal_candidates = [cell for cell in candidates if not cell.get('is_merged', False)]
    for cell in normal_candidates:
        apply_overflow_to_merged_cell(cell, cells, config, table_bbox)

def apply_overflow_to_merged_cell(cell, all_cells, config, table_bbox):
    directions = ['up', 'down']
    random.shuffle(directions)
    cell_height = cell['y2'] - cell['y1']
    
    for direction in directions:
        overflow_height = get_random_overflow_height(cell_height, config)
        height_up = overflow_height if direction == 'up' else 0
        height_down = overflow_height if direction == 'down' else 0
        
        if check_overflow_validity(cell, all_cells, direction, height_up, height_down, config, table_bbox):
            apply_overflow(cell, direction, height_up, height_down, table_bbox)
            return True
    return False

def apply_overflow(cell, direction, height_up, height_down, table_bbox):
    cell['overflow'] = {'direction': direction, 'height_up': height_up, 'height_down': height_down}
    cell['overflow_y1'] = max(cell['y1'] - height_up, table_bbox[1])
    cell['overflow_y2'] = min(cell['y2'] + height_down, table_bbox[3])
    cell['overflow_applied'] = True
    cell['cell_type'] = 'overflow_cell'

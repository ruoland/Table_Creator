import random
from dataset_utils import is_overlapping
from logging_config import table_logger
from dataset_config import TableGenerationConfig

def can_merge(config: TableGenerationConfig):
    if config.enable_horizontal_merge or config.enable_vertical_merge:
        return True
    return False

def create_merged_cell(cells, start_row, start_col, merge_rows, merge_cols, cols, config: TableGenerationConfig):
    # 병합 범위가 테이블을 벗어나지 않는지 확인
    if (start_row + merge_rows > config.total_rows) or (start_col + merge_cols > cols):
        return None  # 병합 범위가 테이블을 벗어나면 None 반환
    
    base_cell_index = start_row * cols + start_col
    end_cell_index = (start_row + merge_rows - 1) * cols + (start_col + merge_cols - 1)
    
    # 인덱스가 유효한지 확인
    if base_cell_index < 0 or end_cell_index < 0 or base_cell_index >= len(cells) or end_cell_index >= len(cells):
        return None  # 유효하지 않은 인덱스면 None 반환
    
    base_cell = cells[base_cell_index]
    end_cell = cells[end_cell_index]

    # 새로운 셀 정보 생성
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
    
    # 병합된 셀의 범위 정보 추가
    new_cell_info['merged_range'] = (start_row, start_col, start_row + merge_rows, start_col + merge_cols)
    config.total_merged_cells = config.total_merged_cells + 1
    return new_cell_info

def manage_merged_cells(cells):
    merged_cells = []
    for cell in cells:
        if cell.get('is_merged', False):
            merged_cells.append(cell)
    return merged_cells

def merge_cells(cells, rows, cols, config):
    #행과 열 자체가 3개 이하인 경우나 병합 설정이 꺼져있는지 여부 확인
    if rows <= 3 or cols <= 3 or not can_merge(config):
        return cells
    
    merged_cells = perform_cell_merging(cells, rows, cols, config)
    return merged_cells

#주변에 병합될 수 있는 셀을 찾습니다
def perform_cell_merging(cells, rows, cols, config):
    if not can_merge(config):
        return cells

    merged_cells = cells.copy()
    merged_areas = []

    for row in range(rows):
        for col in range(cols):
            if (row == 0 and config.table_type in ['header_row', 'header_both']) or \
               (col == 0 and config.table_type in ['header_column', 'header_both']):
                continue  # 헤더 스킵

            if config.enable_horizontal_merge and random.random() < config.horizontal_merge_probability:
                max_merge_cols = min(config.max_horizontal_merge, cols - col)
                if max_merge_cols < 2:
                    continue  # 병합할 수 있는 열이 충분하지 않으면 스킵
                merge_rows = random.randint(2, min(config.max_horizontal_merge, 3))
                merge_cols = random.randint(2, max_merge_cols)
            elif config.enable_vertical_merge and random.random() < config.vertical_merge_probability:
                max_merge_rows = min(config.max_vertical_merge, rows - row)
                if max_merge_rows < 2:
                    continue  # 병합할 수 있는 행이 충분하지 않으면 스킵
                merge_rows = random.randint(2, max_merge_rows)
                merge_cols = random.randint(1, min(2, cols - col))
            else:
                continue

            merge_area = (row, col, row + merge_rows, col + merge_cols)

            if can_merge_and_overflow(merge_area, merged_areas, cells):
                new_cell_info = create_merged_cell(cells, row, col, merge_rows, merge_cols, cols, config)
                if new_cell_info:
                    merged_cells[row * cols + col] = new_cell_info
                    merged_areas.append(merge_area)
                    #print(f"Successfully merged area: {merge_area}")
            else:
                #print(f"Cannot merge area {merge_area} - overlapping or not mergeable")
                pass

    # 병합된 셀 정보 업데이트 및 중복 셀 제거
    #print('Updating merged cell information:', merged_areas)
    for area in merged_areas:
        row_start, col_start, row_end, col_end = area
        base_index = row_start * cols + col_start

        if base_index >= len(merged_cells):
            print(f"Warning: Invalid index for base_cell at {base_index}")
            continue

        base_cell = merged_cells[base_index]
        if base_cell is None:
            print(f"Warning: base_cell is None at {base_index}")
            continue

        base_cell['cell_type'] = 'merged_cell'
        base_cell['is_merged'] = True
        base_cell['merge_rows'] = row_end - row_start
        base_cell['merge_cols'] = col_end - col_start

        # 병합된 영역의 다른 셀들 제거
        for r in range(row_start, row_end):
            for c in range(col_start, col_end):
                if r != row_start or c != col_start:
                    index = r * cols + c
                    if index < len(merged_cells):
                        merged_cells[index] = None

    # None이 아닌 셀과 헤더 셀 모두 유지
    merged_cells = [cell for cell in merged_cells if cell is not None]
    #print('Merged cell information update completed')
    return merged_cells

def is_mergeable(merged_areas, row_start, col_start, row_end, col_end):
    for r in range(row_start, row_end):
        for c in range(col_start, col_end):
            if any((r >= area[0] and r < area[2] and c >= area[1] and c < area[3]) for area in merged_areas):
                return False
    return True

def is_overlapping_with_merged(merge_area, merged_areas):
    current_row, current_col, current_row_end, current_col_end = merge_area

    for area in merged_areas:
        area_row, area_col, area_row_end, area_col_end = area

        horizontal_overlap = (current_col < area_col_end and current_col_end > area_col)
        vertical_overlap = (current_row < area_row_end and current_row_end > area_row)

        if horizontal_overlap and vertical_overlap:
            return True

    return False

def is_overlapping_with_overflow(current_cell, cells):
    current_row = current_cell['row']
    current_col = current_cell['col']
    current_y1 = current_cell['y1']
    current_y2 = current_cell['y2']

    for cell in cells:
        if cell == current_cell or not cell.get('overflow'):
            continue

        cell_row = cell['row']
        cell_col = cell['col']
        cell_y1 = cell.get('overflow_y1', cell['y1'])
        cell_y2 = cell.get('overflow_y2', cell['y2'])

        horizontal_overlap = (current_col == cell_col)
        vertical_overlap = (current_y1 < cell_y2 and current_y2 > cell_y1)

        if horizontal_overlap and vertical_overlap:
            return True

    return False
def can_merge_and_overflow(merge_area, merged_areas, cells):
    row_start, col_start, row_end, col_end = merge_area
    rows = max(cell['row'] for cell in cells) + 1
    cols = max(cell['col'] for cell in cells) + 1

    # 병합 영역 유효성 검사
    if row_end > rows or col_end > cols:
        #print(f"Warning: Invalid merge_area {merge_area}")
        return False

    # 병합 가능 여부 확인
    if not is_mergeable(merged_areas, row_start, col_start, row_end, col_end) or \
       is_overlapping_with_merged(merge_area, merged_areas):
        return False

    # 오버플로우된 셀과의 겹침 확인
    for r in range(row_start, row_end):
        for c in range(col_start, col_end):
            cell = next((cell for cell in cells if cell['row'] == r and cell['col'] == c), None)
            if cell and is_overlapping_with_overflow(cell, cells):
                return False

    return True


#병합 셀 처리 후, 아래 함수 호출
def plan_cell_overflow(table, cells, config: TableGenerationConfig):

    #오버플로우 적용 개수
    overflow_count = 0
    
    #영향 받는 셀들
    affected_cells = set()
    
    #병합 셀과 오버플로 셀 븐석? 왜 있는진 모름
    merged_rows_cols, overflowed_cells = analyze_cells(cells)

    # 모든 셀을 한 번에 처리
    for cell in cells:
        
        #행과 열이 있고 영향받지 않았을 경우
        if (cell['row'], cell['col']) not in affected_cells:
            #오버플로우 셀 처리 시작
            if process_overflow_for_cell(table, cell, cells, affected_cells, config):
                overflow_count += 1
                overflowed_cells.add((cell['row'], cell['col']))

    table_logger.warning(f"오버플로우 계획 완료: 총 {overflow_count}개 셀에 적용")
    return cells
def process_overflow_for_cell(table, cell, all_cells, affected_cells, config):
    row, col = cell['row'], cell['col']
    if not cell.get('is_header', False):
        if cell.get('cell_type') == 'overflow_cell' or (cell.get('cell_type') == 'overflow_cell' and cell.get('is_merged', False)):
            return False

        if random.random() < config.overflow_probability:
            merge_rows = cell.get('merge_rows', 1)
            merge_cols = cell.get('merge_cols', 1)

            possible_directions = ['down'] if row == 0 else ['up'] if row + merge_rows == config.total_rows else ['up', 'down', 'both']
            direction = random.choice(possible_directions)

            overflow_height = get_random_overflow_height(cell, config)
            height_up = overflow_height if direction == 'up' else (overflow_height // 2 if direction == 'both' else 0)
            height_down = overflow_height if direction == 'down' else (overflow_height - height_up if direction == 'both' else 0)

            # 병합된 셀과 오버플로우된 셀 검사 함수
            def check_cells_in_direction(start_row, end_row, start_col, end_col):
                for check_cell in all_cells:
                    check_row = check_cell['row']
                    check_col = check_cell['col']
                    
                    # 병합된 셀 검사
                    if check_cell.get('is_merged'):
                        merged_row_end = check_row + check_cell['merge_rows']
                        merged_col_end = check_col + check_cell['merge_cols']
                        if (start_row < merged_row_end and end_row > check_row and
                            start_col < merged_col_end and end_col > check_col):
                            return True
                    
                    # 오버플로우된 셀 검사
                    if check_cell.get('overflow'):
                        overflow_y1 = check_cell.get('overflow_y1', check_cell['y1'])
                        overflow_y2 = check_cell.get('overflow_y2', check_cell['y2'])
                        if (start_row < check_row + 1 and end_row > check_row and
                            start_col < check_col + 1 and end_col > check_col and
                            start_row * table.row_heights[0] < overflow_y2 and
                            end_row * table.row_heights[0] > overflow_y1):
                            return True
                    
                    # 오버플로우의 영향을 받는 셀 검사
                    if check_cell.get('affected_by_overflow'):
                        if (start_row <= check_row < end_row and
                            start_col <= check_col < end_col):
                            return True
                
                return False

            # 위쪽 오버플로우 검사
            if direction in ['up', 'both']:
                if check_cells_in_direction(row - height_up, row, col, col + merge_cols):
                    return False

            # 아래쪽 오버플로우 검사
            if direction in ['down', 'both']:
                if check_cells_in_direction(row + merge_rows, row + merge_rows + height_down, col, col + merge_cols):
                    return False

            # 테이블 경계 확인
            table_y1, table_y2 = table.table_bbox[1], table.table_bbox[3]
            
            if direction in ['up', 'both']:
                new_y1 = cell['y1'] - height_up
                if new_y1 < table_y1:
                    return False
            
            if direction in ['down', 'both']:
                new_y2 = cell['y2'] + height_down
                if new_y2 > table_y2:
                    return False

            # 오버플로우 정보 부여
            apply_overflow_to_cell(cell, direction, height_up, height_down)
            config.total_overflow_cells = config.total_overflow_cells + 1
            
            # 영향 받는 셀에 표시
            for r in range(row - height_up, row + merge_rows + height_down):
                for c in range(col, col + merge_cols):
                    affected_cell = next((cell for cell in all_cells if cell['row'] == r and cell['col'] == c), None)
                    if affected_cell:
                        affected_cell['affected_by_overflow'] = True
                        affected_cells.add((r, c))

            return True

    return False

def apply_overflow_to_cell(cell, direction, height_up, height_down):
    cell['overflow'] = {
        'direction': direction,
        'height_up': height_up,
        'height_down': height_down
    }
    cell['overflow_y1'] = cell['y1'] - height_up
    cell['overflow_y2'] = cell['y2'] + height_down
    cell['cell_type'] = 'overflow_cell'

def analyze_cells(cells):
    merged_rows_cols = set()
    overflowed_cells = set()
    for cell in cells:
        if cell.get('is_merged', False):
            for r in range(cell['row'], cell['row'] + cell.get('merge_rows', 1)):
                for c in range(cell['col'], cell['col'] + cell.get('merge_cols', 1)):
                    merged_rows_cols.add((r, c))
        if cell.get('overflow'):
            overflowed_cells.add((cell['row'], cell['col']))
    
    table_logger.debug(f"Merged cells: {merged_rows_cols}")
    table_logger.debug(f"Overflowed cells: {overflowed_cells}")
    
    return merged_rows_cols, overflowed_cells


def apply_overflow_to_cell(cell, direction, height_up, height_down):
    cell['overflow'] = {
        'direction': direction,
        'height_up': height_up,
        'height_down': height_down,
    }
    cell['overflow_y1'] = cell['y1'] - height_up if direction in ['up', 'both'] else cell['y1']
    cell['overflow_y2'] = cell['y2'] + height_down if direction in ['down', 'both'] else cell['y2']
    cell['overflow_applied'] = True
    cell['cell_type'] = 'overflow_cell'
def get_random_overflow_height(cell: dict, config: TableGenerationConfig):
    # 최소 높이 계산
    min_height = max(config.min_overflow_height, int(cell['y2'] / 4))
    
    # 최대 높이 계산 (셀 높이의 2/3를 넘지 않도록 조건 추가)
    cell_height = cell['y2'] - cell['y1']  # 현재 셀의 높이
    max_allowed_height = int(cell_height * (2 / 3))  # 셀 높이의 2/3
    max_height = min(min_height + random.randint(1, 5), int(cell['y2'] / 2), max_allowed_height)

    # overflow_height가 일반 셀 크기를 넘어가지 않도록 조정
    if max_height > cell_height:
        max_height = cell_height  # 한 행의 높이를 초과하지 않도록 설정

    # min_height가 max_height보다 클 경우 처리
    if min_height > max_height:
        return max_height  # 또는 min_height, 상황에 따라 선택
    
    return random.randint(min_height, max_height)

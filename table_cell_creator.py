import random
from dataset_utils import is_overlapping
from logging_config import table_logger
def plan_cell_overflow(cells, config):
    if not config.enable_overflow:
        return cells

    overflow_count = 0
    merged_overflow_count = 0
    affected_cells = set()
    cell_dict = {(cell['row'], cell['col']): cell for cell in cells}

    table_logger.warning(f"오버플로우 계획 시작: 총 {len(cells)}개 셀")

    # 셀을 무작위 순서로 처리하되, 병합된 셀을 우선적으로 처리
    shuffled_cells = sorted(cells, key=lambda x: x.get('is_merged', False), reverse=True)
    shuffled_cells = random.sample(shuffled_cells, len(shuffled_cells))

    
    for cell in shuffled_cells:
        #헤더 셀이 아니고, 행과 열을 초과하지 않고, 영향 받는 셀이 아닌 경우
        if (not cell['is_header'] and 
            0 <= cell['row'] < config.total_rows and  
            0 <= cell['col'] < config.total_cols and  
            (cell['row'], cell['col']) not in affected_cells):
            
            if cell['cell_type'] == 'merged_overflow_cell':
                continue
            #확률 계산하기, 만약 병합된 셀인 경우 전용 확률을 사용, 아닌 경우 일반 오버플로 확률 사용
            overflow_prob = config.merged_overflow_probability if cell['is_merged'] else config.overflow_probability

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
                        cell['cell_type'] = 'merged_overflow_cell'
                        table_logger.warning(f"병합 오버플로우 계획: is_merged={cell['is_merged']}direction={direction}, height_up={height_up}, height_down={height_down}")
                        
                        merged_overflow_count += 1
                    else:
                        cell['cell_type'] = 'overflow_cell'
                        overflow_count += 1

                    for r in range(cell['row'], cell['row'] + merge_rows):
                        for c in range(cell['col'], cell['col'] + merge_cols):
                            affected_cells.add((r, c))
                else:
                    table_logger.debug(f"오버플로우 적용 불가: row={cell['row']}, col={cell['col']}")


    table_logger.warning(f"오버플로우 계획 완료: 총 {overflow_count}개 일반 셀, {merged_overflow_count}개 병합 셀에 적용")
    return cells

def is_overlapping_with_merged_cells(new_cell, merged_cells):
    new_row_start = new_cell['row']
    new_col_start = new_cell['col']
    new_row_end = new_row_start + new_cell.get('merge_rows', 1)
    new_col_end = new_col_start + new_cell.get('merge_cols', 1)
    
    for cell in merged_cells:
        if cell.get('is_merged', False) or cell.get('overflow'):
            cell_row_start = cell['row']
            cell_col_start = cell['col']
            cell_row_end = cell_row_start + cell.get('merge_rows', 1)
            cell_col_end = cell_col_start + cell.get('merge_cols', 1)

            # 겹침 조건 확인
            if (new_col_start < cell_col_end and new_col_end > cell_col_start and
                new_row_start < cell_row_end and new_row_end > cell_row_start):
                return True
    return False


def apply_overflow_to_merged_cell(cell, all_cells, config, table_bbox):
    directions = ['up', 'down']
    random.shuffle(directions)  # 방향을 무작위로 섞음

    for direction in directions:
        min_height = min(config.min_overflow_height, config.max_cell_height)
        max_height = max(config.min_overflow_height, config.max_cell_height)
        overflow_height = random.randint(min_height, max_height)

        
        if direction == 'up':
            height_up = min(overflow_height, cell['y1'] - table_bbox[1])
            height_down = 0
        else:  # down
            height_up = 0
            height_down = min(overflow_height, table_bbox[3] - cell['y2'])

        # 오버플로우된 영역 계산
        overflow_y1 = max(cell['y1'] - height_up, table_bbox[1])
        overflow_y2 = min(cell['y2'] + height_down, table_bbox[3])

        # 임시 오버플로우 셀 생성
        temp_overflow_cell = {
            'x1': cell['x1'], 
            'y1': overflow_y1,
            'x2': cell['x2'], 
            'y2': overflow_y2,
            'row': cell['row'],  # row 추가
            'col': cell['col'],  # col 추가
            'is_merged': cell.get('is_merged', False),  # 병합 여부
            'is_header': cell.get('is_header', False),  # 헤더 여부
            'original_height': cell.get('original_height'),  # 원래 높이
            'merge_rows': cell.get('merge_rows', 1),  # 병합된 행 수
            'merge_cols': cell.get('merge_cols', 1),  # 병합된 열 수
            'cell_type': cell.get('cell_type', 'normal_cell'),  # 셀 타입
        }

        # 주변 셀과의 겹침 확인 (병합된 셀 포함)
        if not is_overlapping_with_nearby_cells(cell, all_cells, direction, height_up, height_down, config) and \
           not is_overlapping_with_merged_cells(temp_overflow_cell, all_cells):
            # 겹치지 않는 경우 오버플로우 적용
            cell['overflow'] = {
                'direction': direction,
                'height_up': height_up,
                'height_down': height_down
            }
            cell['overflow_y1'] = overflow_y1
            cell['overflow_y2'] = overflow_y2
            cell['overflow_applied'] = True
            cell['cell_type'] = 'merged_overflow_cell'
            return True  # 오버플로우 적용 성공

    return False  # 모든 방향에 대해 오버플로우 적용 실패
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



def validate_cell_coordinates(cells, table_bbox):
    for cell in cells:
        # x 좌표 조정
        cell['x1'] = max(table_bbox[0], min(cell['x1'], table_bbox[2] - 1))
        cell['x2'] = min(table_bbox[2], max(cell['x2'], cell['x1'] + 1))
        
        # y 좌표 조정
        cell['y1'] = max(table_bbox[1], min(cell['y1'], table_bbox[3] - 1))
        cell['y2'] = min(table_bbox[3], max(cell['y2'], cell['y1'] + 1))
        
        # 최소 크기 보장
        if cell['x2'] - cell['x1'] < 1:
            cell['x2'] = cell['x1'] + 1
        if cell['y2'] - cell['y1'] < 1:
            cell['y2'] = cell['y1'] + 1
        
    return cells
def adjust_cell_positions(cells, config, table_bbox):
    if not config.enable_overflow:
        return cells
    
    table_logger.warning(f"셀 위치 조정 시작: 총 {len(cells)}개 셀")
    log_cell_coordinates(cells, "Start of adjust_cell_positions")
    adjusted_count = 0
    
    for cell in cells:
        original_x1, original_x2 = cell['x1'], cell['x2']
        original_y1, original_y2 = cell['y1'], cell['y2']
        
        # 기본 x1 및 x2 좌표 검증 추가 
        cell['x1'] = max(cell['x1'], table_bbox[0])
        cell['x2'] = min(cell['x2'], table_bbox[2])

        # x1이 x2보다 크거나 같은 경우 조정 
        if cell['x1'] >= cell['x2']:
            middle_x = (cell['x1'] + cell['x2']) / 2 
            cell['x1'] = max(middle_x - 1, table_bbox[0]) 
            cell['x2'] = min(middle_x + 1, table_bbox[2]) 
        
        if cell['x1'] != original_x1 or cell['x2'] != original_x2:
            table_logger.warning(f"X 좌표 조정: row={cell['row']}, col={cell['col']}, x1: {original_x1} -> {cell['x1']}, x2: {original_x2} -> {cell['x2']}")

        if cell.get('overflow') and cell.get('overflow_applied', False):
            direction = cell['overflow']['direction']
            height_up = cell['overflow']['height_up']
            height_down = cell['overflow']['height_down']
            
            # 병합된 셀 여부 확인
            is_merged = cell.get('is_merged', False)
            merge_rows = cell.get('merge_rows', 1)
            merge_cols = cell.get('merge_cols', 1)
            
            if direction in ['up', 'both']:
                cell['overflow_y1'] = max(cell['y1'] - height_up, table_bbox[1])
            if direction in ['down', 'both']:
                cell['overflow_y2'] = min(cell['y2'] + height_down, table_bbox[3])
            
            # 실제로 오버플로우가 적용되었는지 다시 확인
            if cell['overflow_y1'] == original_y1 and cell['overflow_y2'] == original_y2:
                table_logger.warning(f"오버플로우 취소: row={cell['row']}, col={cell['col']}, 변화 없음")
                cell.pop('overflow', None)
                cell.pop('overflow_y1', None)
                cell.pop('overflow_y2', None)
                cell.pop('overflow_applied', None)
                if cell['is_merged']:
                    cell['cell_type'] = 'merged_cell'
                else:
                    cell['cell_type'] = 'cell'
            else:
                table_logger.warning(f"오버플로우 적용: row={cell['row']}, col={cell['col']}, direction={direction}, y1: {original_y1} -> {cell['overflow_y1']}, y2: {original_y2} -> {cell['overflow_y2']}")
                adjusted_count += 1 

    for cell in cells:
        if 'overflow_y1' in cell:
            original_y1 = cell['overflow_y1']
            cell['overflow_y1'] = max(min(cell['overflow_y1'], cell['y2']-1), table_bbox[1])
            if cell['overflow_y1'] != original_y1:
                table_logger.warning(f"오버플로우 y1 조정: row={cell['row']}, col={cell['col']}, y1: {original_y1} -> {cell['overflow_y1']}")
        
        if 'overflow_y2' in cell:
            original_y2 = cell['overflow_y2']
            cell['overflow_y2'] = min(max(cell['overflow_y2'], cell['y1']+1), table_bbox[3])
            if cell['overflow_y2'] != original_y2:
                table_logger.warning(f"오버플로우 y2 조정: row={cell['row']}, col={cell['col']}, y2: {original_y2} -> {cell['overflow_y2']}")
        
        # 기본 y 좌표 검증 
        original_y1, original_y2 = cell['y1'], cell['y2']
        cell['y1'] = max(cell['y1'], table_bbox[1])
        cell['y2'] = min(cell['y2'], table_bbox[3])
        
        # y 좌표 조정 
        if cell['y1'] >= cell['y2']:
            middle_y = (cell['y1'] + cell['y2']) / 2 
            cell['y1'] = max(middle_y - 1, table_bbox[1]) 
            cell['y2'] = min(middle_y + 1, table_bbox[3]) 
        
        if cell['y1'] != original_y1 or cell['y2'] != original_y2:
            table_logger.warning(f"Y 좌표 조정: row={cell['row']}, col={cell['col']}, y1: {original_y1} -> {cell['y1']}, y2: {original_y2} -> {cell['y2']}")

        # 최종 검증 
        cell['x1'] = max(min(cell['x1'], table_bbox[2]), table_bbox[0]) 
        cell['x2'] = max(min(cell['x2'], table_bbox[2]), table_bbox[0]) 
        cell['y1'] = max(min(cell['y1'], table_bbox[3]), table_bbox[1]) 
        cell['y2'] = max(min(cell['y2'], table_bbox[3]), table_bbox[1]) 

        # 최소 크기 확인 
        if (cell['x2'] - cell['x1']) < 10 or (cell['y2'] - cell['y1']) < 10:
            table_logger.warning(f"Invalid size after adjustment: {cell}")

    log_cell_coordinates(cells, "End of adjust_cell_positions")
    table_logger.warning(f"셀 위치 조정 완료: {adjusted_count}개 셀의 위치가 조정됨")
    return cells



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
def merge_cells(cells, rows, cols, config, table_bbox):
    table_logger.debug(f"merge_cells 시작: 행 {rows}, 열 {cols}")
    
    if rows <= 3 or cols <= 3 or not can_merge(config):
        table_logger.debug("테이블이 너무 작거나 셀 병합이 비활성화되어 병합을 수행하지 않습니다.")
        return cells
    if not cells:
        table_logger.error("병합할 셀이 없습니다.")
        return []  # 빈 리스트 반환

    merged_cells = perform_cell_merging(cells, rows, cols, config)
    
    # 병합된 셀들 중에서 오버플로우 후보 선정
    overflow_candidates = select_overflow_candidates(merged_cells, config)
    
    # 오버플로우 적용
    apply_overflow_to_cells(overflow_candidates, merged_cells, config, table_bbox)

    # 셀 ID 재할당
    merged_cells = reassign_cell_ids(merged_cells)

    log_cell_coordinates(merged_cells, "End of merge_cells")
    table_logger.debug(f"merge_cells 종료: 병합된 셀 수 {len([c for c in merged_cells if c['is_merged']])}, 남은 셀 수 {len(merged_cells)}")
    return merged_cells
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
    log_cell_coordinates(merged_cells, "Start of merge_cells")

    if config.enable_horizontal_merge:
        for row in range(rows):
            col = 0 
            while col < cols - 3:
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
            row = 0
            while row < rows - 2:
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

    # None이 아닌 셀만 유지
    return [cell for cell in merged_cells if cell is not None]

def select_overflow_candidates(merged_cells, config):
    return [cell for cell in merged_cells if cell['is_merged'] and random.random() < config.overflow_probability]

def apply_overflow_to_cells(candidates, cells, config, table_bbox):
    for cell in candidates:
        apply_overflow_to_merged_cell(cell, cells, config, table_bbox)

def is_overlapping_with_merged_or_overflow(new_area, merged_areas, cells):
    row_start, col_start, row_end, col_end = new_area
    
    # 기존 병합 영역과의 겹침 확인
    if any(is_overlapping(new_area, area) for area in merged_areas):
        return True
    
    # 오버플로우된 셀과의 겹침 확인
    for cell in cells:
        if cell and (cell.get('overflow') or cell.get('cell_type') == 'merged_overflow_cell'):
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

def reassign_cell_ids(cells):
    row_map = {}
    col_map = {}
    new_row = 0
    new_col = 0

    for cell in cells:
        if cell['row'] not in row_map:
            row_map[cell['row']] = new_row
            new_row += 1
        if cell['col'] not in col_map:
            col_map[cell['col']] = new_col
            new_col += 1
        
        cell['row'] = row_map[cell['row']]
        cell['col'] = col_map[cell['col']]

    return cells


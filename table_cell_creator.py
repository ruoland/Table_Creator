import random
import numpy as np
import cv2
from dataset_utils import is_overlapping, validate_cell
from dataset_config import TableGenerationConfig, CELL_CATEGORY_ID, TABLE_CATEGORY_ID, COLUMN_CATEGORY_ID, ROW_CATEGORY_ID
from logging_config import table_logger, get_memory_handler
def plan_cell_overflow(cells, config):
    if not config.enable_overflow:
        return cells

    rows = max(cell['row'] for cell in cells) + 1
    cols = max(cell['col'] for cell in cells) + 1

    overflow_count = 0
    affected_cells = set()

    for cell in cells:
        if (not cell['is_header'] and 
            1 <= cell['row'] < rows - 1 and  
            1 <= cell['col'] < cols - 1 and  
            random.random() < config.overflow_probability and
            (cell['row'], cell['col']) not in affected_cells):  
            
            overflow_height = random.randint(config.min_overflow_height, config.max_overflow_height)
            direction = random.choice(['up', 'down', 'both'])
            
            if direction == 'up':
                height_up = overflow_height
                height_down = 0
            elif direction == 'down':
                height_up = 0
                height_down = overflow_height
            else:  
                height_up = overflow_height // 2
                height_down = overflow_height - height_up
                
            cell['overflow'] = {
                'direction': direction,
                'height_up': height_up,
                'height_down': height_down,
                'affected_cells': []
            }

            # 오버플로우 처리
            for r in range(cell['row']-1, -1, -1):
                affected_cell = next((c for c in cells if c['row'] == r and c['col'] == cell['col']), None)
                if affected_cell:
                    cell['overflow']['affected_cells'].append(affected_cell)
                    affected_cells.add((affected_cell['row'], affected_cell['col']))
                    if cell['y1'] - cell['overflow']['height_up'] >= affected_cell['y1']:
                        break

            for r in range(cell['row']+1, rows):
                affected_cell = next((c for c in cells if c['row'] == r and c['col'] == cell['col']), None)
                if affected_cell:
                    cell['overflow']['affected_cells'].append(affected_cell)
                    affected_cells.add((affected_cell['row'], affected_cell['col']))
                    if cell['y2'] + cell['overflow']['height_down'] <= affected_cell['y2']:
                        break

            # 오버플로우된 셀의 타입 업데이트
            if cell.get('overflow'):
                        if cell['cell_type'] == 'merged_cell':
                            cell['cell_type'] = 'merged_overflow_cell'
                        else:
                            cell['cell_type'] = 'overflow_cell'
            overflow_count += 1

    table_logger.info(f"오버플로우 계획: 총 {overflow_count}개 셀에 적용")
    return cells
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
        
        table_logger.debug(f"Validated cell: {cell}")
    return cells

# create_table 함수의 마지막에 추가
def adjust_cell_positions(cells, config, table_bbox):
    if not config.enable_overflow:
        return cells
    log_cell_coordinates(cells, "Start of adjust_cell_positions")
    adjusted_count = 0
    for cell in cells:
        # 기본 x1 및 x2 좌표 검증 추가 
        cell['x1'] = max(cell['x1'], table_bbox[0])
        cell['x2'] = min(cell['x2'], table_bbox[2])

        # x1이 x2보다 크거나 같은 경우 조정 
        if cell['x1'] >= cell['x2']:
            middle_x = (cell['x1'] + cell['x2']) / 2 
            cell['x1'] = max(middle_x - 1 ,table_bbox[0]) 
            cell['x2'] = min(middle_x + 1 ,table_bbox[2]) 

        if cell.get('overflow'):
            direction=cell ['overflow']['direction']
            height_up=cell ['overflow']['height_up']
            height_down=cell ['overflow']['height_down']
            
            if direction in ['up', 'both']:
                cell ['overflow_y1']=max(cell ['y1']-height_up ,table_bbox[1])
            if direction in ['down', 'both']:
                cell ['overflow_y2']=min(cell ['y2']+height_down ,table_bbox[3])
            
            for affected_cell in cell ['overflow']['affected_cells']:
                if affected_cell ['row']<cell ['row'] and direction in ['up', 'both']:
                    affected_cell ['overflow_y2']=max(cell ['overflow_y1'], affected_cell ['y1'])
                    affected_cell ['is_affected_by_overflow']=True 
                elif affected_cell ['row']>cell ['row'] and direction in ['down', 'both']:
                    affected_cell ['overflow_y1']=min(cell ['overflow_y2'], affected_cell ['y2'])
                    affected_cell ['is_affected_by_overflow']=True 
                
                adjusted_count+=1 

    for cell in cells:
        if'overflow_y1'in cell:
            cell ['overflow_y1']=max(min(cell ['overflow_y1'],cell ['y2']-1),table_bbox[1])
        
        if'overflow_y2'in cell:
            cell ['overflow_y2']=min(max(cell ['overflow_y2'],cell ['y1']+1),table_bbox[3])
        
        # 기본 y 좌표 검증 
        cell ['y1']=max(cell ['y1'],table_bbox[1])
        cell ['y2']=min(cell ['y2'],table_bbox[3])
        
        # y 좌표 조정 
        if cell ['y1']>=cell ['y2']:
            middle_y=(cell ['y1']+cell ['y2'])/2 
            cell ['y1']=max(middle_y-1 ,table_bbox[1]) 
            cell ['y2']=min(middle_y+1 ,table_bbox[3]) 

        # 최종 검증 
        cell ['x1']=max(min(cell ['x1'],table_bbox[2]),table_bbox[0]) 
        cell ['x2']=max(min(cell ['x2'],table_bbox[2]),table_bbox[0]) 
        cell ['y1']=max(min(cell ['y1'],table_bbox[3]),table_bbox[1]) 
        cell ['y2']=max(min(cell ['y2'],table_bbox[3]),table_bbox[1]) 

        # 최소 크기 확인 
        if (cell['x2']-cell['x1'])< 10 or (cell['y2']-cell['y1'])<10:
            table_logger.warning(f"Invalid size after adjustment: {cell}")
    log_cell_coordinates(cells, "End of adjust_cell_positions")
    table_logger.info(f"셀 위치 조정: {adjusted_count}개 셀의 위치가 조정됨")
    return cells
def log_cell_coordinates(cells, stage):
    table_logger.debug(f"=== Cell coordinates at {stage} ===")
    for cell in cells:
        table_logger.debug(f"Cell: row={cell['row']}, col={cell['col']}, x1={cell['x1']}, y1={cell['y1']}, x2={cell['x2']}, y2={cell['y2']}")
    table_logger.debug("=" * 50)
def merge_cells(cells, rows, cols, config):
    table_logger.debug(f"merge_cells 시작: 행 {rows}, 열 {cols}")
    if rows <= 3 or cols <= 3 or not config.enable_cell_merging:
        table_logger.debug("테이블이 너무 작거나 셀 병합이 비활성화되어 병합을 수행하지 않습니다.")
        return cells
    if not cells:
        table_logger.error("병합할 셀이 없습니다.")
        return []  # 빈 리스트 반환

    merged_cells = cells.copy()
    merged_areas = []
    log_cell_coordinates(merged_cells, "Start of merge_cells")

    # 수평 병합 
    if config.enable_horizontal_merge:
        for row in range(rows):
            col = 0 
            while col < cols - 3:
                if random.random() < config.horizontal_merge_probability:
                    merge_cols = random.randint(2, min(config.max_horizontal_merge, cols - col))
                    merge_area = (row, col, row + 3, col + merge_cols) 
                    if not any(is_overlapping(merge_area, area) for area in merged_areas):
                        new_cell_info = create_merged_cell(merged_cells, row, col, 3, merge_cols, cols, config) 
                        if new_cell_info:
                            merged_cells[row * cols + col] = new_cell_info 
                            merged_areas.append(merge_area) 
                            col += merge_cols 
                            continue 
                col += 3 

    # 수직 병합 
    if config.enable_vertical_merge:
        for col in range(cols):
            row = 0
            while row < rows - 2:  # 최소 2행이 남아있을 때까지만 병합 시도
                if random.random() < config.vertical_merge_probability:
                    max_merge = min(config.max_vertical_merge, rows - row)
                    if max_merge < 2:
                        break
                    merge_rows = random.randint(2, max_merge)  # 최소 2행부터 병합 가능하도록 변경
                    merge_area = (row, col, row + merge_rows, col + 1)
                    if not any(is_overlapping(merge_area, area) for area in merged_areas):
                        new_cell_info = create_merged_cell(merged_cells, row, col, merge_rows, 1, cols, config)
                        if new_cell_info:
                            merged_cells[row * cols + col] = new_cell_info
                            merged_areas.append(merge_area)
                            row += merge_rows
                            continue
                row += 1

    # 병합된 셀 정보 업데이트
    # 병합된 셀 정보 업데이트
    # 병합된 셀 정보 업데이트
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
    cells = [cell for cell in merged_cells if cell is not None]

    # 셀 ID 재할당
    cells = reassign_cell_ids(cells)

    log_cell_coordinates(cells, "End of merge_cells")
    table_logger.debug(f"merge_cells 종료: 병합된 셀 수 {len(merged_areas)}, 남은 셀 수 {len(cells)}")
    return cells


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

def create_merged_cell(cells, start_row, start_col, merge_rows, merge_cols, cols, config):
    total_rows = len(cells) // cols
    
    # 병합 범위가 테이블을 벗어나지 않는지 확인
    if (start_row + merge_rows > total_rows) or (start_col + merge_cols > cols):
        return None  # 병합 범위가 테이블을 벗어나면 None 반환
    
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

def adjust_probabilities(rows, cols):
    total_cells = rows * cols
    merge_prob = min(0.2, 20 / total_cells)  # 큰 테이블에서는 병합 확률을 낮춤
    overflow_prob = min(0.1, 10 / total_cells)  # 큰 테이블에서는 오버플로우 확률을 낮춤
    return merge_prob, overflow_prob
import random
import numpy as np
import cv2
from dataset_utils import is_overlapping, validate_cell
from dataset_config import TableGenerationConfig, CELL_CATEGORY_ID, TABLE_CATEGORY_ID, COLUMN_CATEGORY_ID, ROW_CATEGORY_ID, MERGED_CELL_CATEGORY_ID, OVERFLOW_CELL_CATEGORY_ID, MERGED_OVERFLOW_CELL_CATEGORY_ID
from logging_config import table_logger, get_memory_handler
from table_cell_creator import *

def create_table(image_width, image_height, margins, title_height, config: TableGenerationConfig):

    memory_handler = get_memory_handler()
    table_logger.addHandler(memory_handler)
    table_type = random.choices(['simple', 'medium', 'complex'], 
                                weights=[config.simple_table_ratio, 
                                        config.medium_table_ratio, 
                                        config.complex_table_ratio])[0]
    try:
        table_logger.debug(f"create_table 시작: 이미지 크기 {image_width}x{image_height}")
        margin_left, margin_top, margin_right, margin_bottom = map(int, margins)
        title_height = int(title_height)
        
        # 갭 설정
        gap = random.randint(config.min_cell_gap, config.max_cell_gap) if config.enable_cell_gap else 0
        if table_type == 'simple':
            max_rows = config.max_rows_simple
            max_cols = config.max_cols_simple
        elif table_type == 'medium':
            max_rows = config.max_rows_medium
            max_cols = config.max_cols_medium
        elif table_type == 'complex':
            max_rows = config.max_rows_complex
            max_cols = config.max_cols_complex
        # 테이블 크기 계산 (갭 포함)
        table_width = max(config.min_table_width, image_width - margin_left - margin_right)
        table_height = max(config.min_table_height, image_height - margin_top - margin_bottom - title_height)

        # 행과 열 수 계산 (갭 고려), 최소 2개의 셀 보장
        cols = max(2, min(max_cols, (table_width + gap) // (config.min_cell_width + gap)))
        rows = max(2, min(max_rows, (table_height + gap) // (config.min_cell_height + gap)))
        
        # 만약 행이나 열이 1인 경우, 다른 쪽을 2로 설정
        if rows == 1:
            cols = max(2, cols)
        elif cols == 1:
            rows = max(2, rows)
        
        # TableGenerationConfig에 total_rows와 total_cols 설정
        config.total_rows = rows
        config.total_cols = cols

        table_logger.info(f"테이블 크기: {table_width}x{table_height}, 행 수: {rows}, 열 수: {cols}, 갭: {gap}")

        # 셀 크기 계산 (갭 제외)
        available_width = table_width - (cols - 1) * gap
        available_height = table_height - (rows - 1) * gap
        min_cell_width = max(1, available_width // cols)
        min_cell_height = max(1, available_height // rows)

        # 헤더 행과 열의 크기 계산
        header_row_height = int(min_cell_height * config.header_row_height_factor)
        header_col_width = int(min_cell_width * config.header_col_width_factor)

        col_widths = [min_cell_width] * cols
        row_heights = [min_cell_height] * rows

        # 헤더 행과 열에 대해 다른 크기 적용
        if config.table_type in ['header_row', 'header_both']:
            row_heights[0] = header_row_height
        if config.table_type in ['header_column', 'header_both']:
            col_widths[0] = header_col_width

        # 남은 공간 분배
        extra_width = available_width - sum(col_widths)
        extra_height = available_height - sum(row_heights)

        for i in range(extra_width):
            col_widths[i % cols] += 1
        for i in range(extra_height):
            row_heights[i % rows] += 1
        target_ratios = config.dataset_counter.get_target_ratios()
            # 셀 타입 결정을 위한 누적 확률 계산
        cell_types = list(target_ratios.keys())
        cell_probs = list(target_ratios.values())
        cumulative_probs = [sum(cell_probs[:i+1]) for i in range(len(cell_probs))]

        cells = []
        y = margin_top + title_height

        for row in range(rows):
            x = margin_left
            for col in range(cols):
                is_header = (config.table_type in ['header_row', 'header_both'] and row == 0) or \
                            (config.table_type in ['header_column', 'header_both'] and col == 0)
                is_gray = random.random() < config.gray_cell_probability

                # 셀 타입 결정
                if config.enable_cell_merging:  # 셀 병합 기능이 활성화된 경우에만 다양한 셀 타입 사용
                    if not is_header:
                        rand_val = random.random()
                        cell_type = next(ct for ct, cp in zip(cell_types, cumulative_probs) if rand_val <= cp)
                    else:
                        cell_type = 'normal_cell'  # 헤더는 항상 normal_cell로 처리
                else:
                    cell_type = 'normal_cell'  # 셀 병합 기능이 비활성화된 경우 모든 셀을 normal_cell로 처리

                cell = {
                    'x1': x,
                    'y1': y,
                    'x2': x + col_widths[col],
                    'y2': y + row_heights[row],
                    'row': row,
                    'col': col,
                    'is_merged': False,  # 초기에는 모든 셀을 병합되지 않은 상태로 설정
                    'is_header': is_header,
                    'is_gray': is_gray,
                    'original_height': row_heights[row],
                    'overflow': None,
                    'cell_type': cell_type
                }
                if cell['y1'] >= cell['y2']:
                    table_logger.warning(f"Invalid cell coordinates detected: {cell}")
                    cell['y2'] = cell['y1'] + 1  # 최소 1픽셀 높이 보장

                cells.append(cell)
                x += col_widths[col] + gap
            y += row_heights[row] + gap

        log_cell_coordinates(cells, "Initial cell creation")
        # 테이블의 전체 경계 상자 계산
        table_bbox = [
            margin_left,
            margin_top + title_height,
            margin_left + table_width,
            margin_top + title_height + table_height
        ]

        table_logger.info(f"초기 셀 생성 완료. 총 셀 수: {len(cells)}")
        cells = validate_all_cells(cells, table_bbox, "cell creation")
        log_cell_coordinates(cells, "After initial validation")

        config.horizontal_merge_probability, config.vertical_merge_probability = adjust_probabilities(rows, cols)

        is_table_rounded = config.enable_rounded_table_corners and random.random() < config.rounded_table_corner_probability
        table_corner_radius = random.randint(config.min_table_corner_radius, config.max_table_corner_radius) if is_table_rounded else 0
        # 셀 병합 (옵션)
        if config.enable_cell_merging and len(cells) > 2:
            cells = merge_cells(cells, rows, cols, config)
            table_logger.info(f"셀 병합 후 셀 수: {len(cells)}")
            log_cell_coordinates(cells, "After cell merging")
        cells = validate_all_cells(cells, table_bbox, "cell merging")

        # 오버플로우 계획
        cells = plan_cell_overflow(cells, config)
        table_logger.info(f"오버플로우 계획 후 셀 수: {len(cells)}")
        log_cell_coordinates(cells, "After overflow planning")
        config.overflow_probability = adjust_probabilities(rows, cols)[1]

        # 셀 위치 조정 전 로깅 추가
        table_logger.info(f"셀 위치 조정 전 셀 수: {len(cells)}")
        log_cell_coordinates(cells, "Before cell position adjustment")
        
        cells = adjust_cell_positions(cells, config, table_bbox)

        table_logger.info(f"셀 위치 조정 후 셀 수: {len(cells)}")
        log_cell_coordinates(cells, "After cell position adjustment")
            
        # 최종 검증: 셀이 2개 미만인 경우 처리
        if len(cells) < 2:
            table_logger.warning("셀 수가 2개 미만입니다. 테이블을 2x1로 재구성합니다.")
            cells = [
                {'x1': margin_left, 'y1': margin_top + title_height, 'x2': margin_left + table_width // 2,
                    'y2': margin_top + title_height + table_height,
                    'row': 0, 'col': 0, 'is_merged': False, 'is_header': False,
                    'original_height': table_height, 'overflow': None},
                {'x1': margin_left + table_width // 2, 'y1': margin_top + title_height,
                    'x2': margin_left + table_width,
                    'y2': margin_top + title_height + table_height,
                    'row': 0, 'col': 1, 'is_merged': False, 'is_header': False,
                    'original_height': table_height, 'overflow': None}
            ]
        cells = validate_cell_coordinates(cells, table_bbox)
        log_cell_coordinates(cells, "Final validation")
        config.dataset_counter.update_counts(cells)
        table_logger.debug(f"create_table 종료: 생성된 셀 수 {len(cells)}")
        return cells, table_bbox, is_table_rounded, table_corner_radius    
    except Exception as e:

        table_logger.error(f"테이블 생성 중 오류 발생: {str(e)}", exc_info=True)
        raise
    finally:
        # 메모리 핸들러의 내용을 모두 출력하고 로거에서 제거
        memory_handler.close()
        table_logger.removeHandler(memory_handler)

def validate_all_cells(cells, table_bbox, stage_name):
    for cell in cells:
        if cell['x1'] >= cell['x2'] or cell['y1'] >= cell['y2']:
            table_logger.warning(f"Invalid cell coordinates detected after {stage_name}: {cell}")
        if cell['x1'] < table_bbox[0] or cell['x2'] > table_bbox[2] or cell['y1'] < table_bbox[1] or cell['y2'] > table_bbox[3]:
            table_logger.warning(f"Cell coordinates out of table bounds after {stage_name}: {cell}")
    return cells


def generate_coco_annotations(cells, table_bbox, image_id, config):
    table_logger.debug(f"generate_coco_annotations 시작: 이미지 ID {image_id}, 셀 수 {len(cells)}")
    coco_annotations = []
    annotation_id = 1

    def create_annotation(bbox, category_id, category_name, attributes=None):
        x1, y1, x2, y2 = bbox
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        if width > 0 and height > 0:
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0,
                "category_name": category_name
            }
            if attributes:
                annotation["attributes"] = attributes
            return annotation
        table_logger.warning(f"유효하지 않은 bbox: {bbox} for {category_name}")
        return None

    # 오버플로우 영향을 받는 셀 식별
    affected_cells = set()
    for cell in cells:
        if cell.get('overflow'):
            for affected_cell in cell['overflow'].get('affected_cells', []):
                affected_cells.add((affected_cell['row'], affected_cell['col']))

    # 셀의 실제 행과 열 ID 수집
    cell_rows = sorted(set(cell['row'] for cell in cells))
    cell_cols = sorted(set(cell['col'] for cell in cells))

    # 행 어노테이션 생성
    for i, row in enumerate(cell_rows):
        row_cells = [cell for cell in cells if cell['row'] == row]
        y1 = min(cell['y1'] for cell in row_cells)
        y2 = max(cell['y2'] for cell in row_cells)
        row_coords = [table_bbox[0], y1, table_bbox[2], y2]
        is_header = i == 0 and config.table_type in ['header_row', 'header_both']
        row_annotation = create_annotation(row_coords, ROW_CATEGORY_ID, "row", {
            "row_id": i,
            "is_header": is_header,
            "y1": y1,
            "y2": y2
        })
        if row_annotation:
            coco_annotations.append(row_annotation)
            annotation_id += 1

    # 열 어노테이션 생성
    for i, col in enumerate(cell_cols):
        col_cells = [cell for cell in cells if cell['col'] == col]
        x1 = min(cell['x1'] for cell in col_cells)
        x2 = max(cell['x2'] for cell in col_cells)
        col_coords = [x1, table_bbox[1], x2, table_bbox[3]]
        is_header = i == 0 and config.table_type in ['header_column', 'header_both']
        col_annotation = create_annotation(col_coords, COLUMN_CATEGORY_ID, "column", {
            "column_id": i,
            "is_header": is_header,
            "x1": x1,
            "x2": x2
        })
        if col_annotation:
            coco_annotations.append(col_annotation)
            annotation_id += 1

# 셀 어노테이션 생성
    for cell_info in cells:
        if cell_info is None:
            table_logger.warning(f"None 셀 발견: image_id {image_id}")
            continue

        x1, y1, x2, y2 = cell_info.get('x1'), cell_info.get('y1'), cell_info.get('x2'), cell_info.get('y2')
        if any(coord is None for coord in [x1, y1, x2, y2]):
            table_logger.warning(f"유효하지 않은 좌표를 가진 셀 발견: image_id {image_id}, cell_info: {cell_info}")
            continue

        overflow = cell_info.get('overflow', {})
        if overflow:
            y1 = min(y1, cell_info.get('overflow_y1', y1))
            y2 = max(y2, cell_info.get('overflow_y2', y2))

        coords = [x1, y1, x2, y2]
        
        is_affected = (cell_info['row'], cell_info['col']) in affected_cells
        
        attributes = {
            "is_header": cell_info.get('is_header', False),
            "is_merged": cell_info.get('is_merged', False),
            "has_overflow": bool(overflow),
            "overflow_direction": overflow.get('direction', 'none') if overflow else 'none',
            "is_affected_by_overflow": is_affected,
            "row": cell_info['row'],
            "col": cell_info['col'],
            "original_height": cell_info.get('original_height'),
            "is_gray": cell_info.get('is_gray', False),
            "row_id": cell_rows.index(cell_info['row']),
            "col_id": cell_cols.index(cell_info['col']),
            "row_span": cell_info.get('merge_rows', 1),
            "col_span": cell_info.get('merge_cols', 1),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        }
        
        # 셀 카테고리 결정
        cell_type = cell_info.get('cell_type', 'normal_cell')
        if cell_info.get('is_merged', False):
            if overflow:
                category_id = MERGED_OVERFLOW_CELL_CATEGORY_ID
                category_name = "merged_overflow_cell"
            else:
                category_id = MERGED_CELL_CATEGORY_ID
                category_name = "merged_cell"
        elif overflow:
            category_id = OVERFLOW_CELL_CATEGORY_ID
            category_name = "overflow_cell"
        else:
            category_id = CELL_CATEGORY_ID
        category_name = "cell"


        annotation = create_annotation(coords, category_id, category_name, attributes)
        if annotation:
            coco_annotations.append(annotation)
            annotation_id += 1

    # 테이블 어노테이션 생성
    table_attributes = {
        "table_type": config.table_type,
        "total_rows": len(cell_rows),  # config.total_rows 대신 실제 행 수 사용
        "total_cols": len(cell_cols),  # config.total_cols 대신 실제 열 수 사용
        "has_outer_border": config.enable_outer_border,
        "has_rounded_corners": config.enable_rounded_corners,
        "has_cell_gap": config.enable_cell_gap,
        "has_overflow": config.enable_overflow,
        "has_merged_cells": config.enable_cell_merging,
        "has_gray_cells": config.enable_gray_cells,
        "x1": table_bbox[0],
        "y1": table_bbox[1],
        "x2": table_bbox[2],
        "y2": table_bbox[3],
        "actual_rows": len(cell_rows),
        "actual_cols": len(cell_cols)
    }
    table_annotation = create_annotation(table_bbox, TABLE_CATEGORY_ID, "table", table_attributes)
    if table_annotation:
        coco_annotations.append(table_annotation)
        annotation_id += 1
    table_logger.info(f"generate_coco_annotations 종료: 이미지 ID {image_id}, 생성된 어노테이션 수 {len(coco_annotations)}")
    table_logger.info(f"테이블 구조: {len(cell_rows)}행 x {len(cell_cols)}열")
    table_logger.info(f"원래 의도된 구조: {config.total_rows}행 x {config.total_cols}열")
    table_logger.info(f"행 불일치: {abs(len(cell_rows) - config.total_rows)}, 열 불일치: {abs(len(cell_cols) - config.total_cols)}")

    return coco_annotations 
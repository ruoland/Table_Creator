import random
import numpy as np
from dataset_config import TableGenerationConfig, CELL_CATEGORY_ID, TABLE_CATEGORY_ID, COLUMN_CATEGORY_ID, ROW_CATEGORY_ID, MERGED_CELL_CATEGORY_ID, OVERFLOW_CELL_CATEGORY_ID, HEADER_ROW_CATEGORY_ID, HEADER_COLUMN_CATEGORY_ID
from logging_config import table_logger, get_memory_handler
from table_cell_creator import *
from dataset_utils import adjust_cell_positions, validate_all_cells
class Table:
    def __init__(self, config: TableGenerationConfig):
        self.config = config
        self.rows = 0
        self.cols = 0
        self.table_width = 0
        self.table_height = 0
        self.cells = []
        self.merged_cells = []  # 병합된 셀 리스트
        self.overflow_cells = []  # 오버플로우 셀 리스트
        self.merged_cells = []
        self.overflow_cells = []
        self.col_widths = []
        self.row_heights = []
        self.table_bbox = []
        self.is_rounded = False
        self.corner_radius = 0
        self.gap = 0
        
        # 여백 정보 추가
        self.margin_left = 0
        self.margin_top = 0
        self.margin_right = 0
        self.margin_bottom = 0
        self.title_height = 0
        
        self.image_width = 0
        self.image_height = 0

    def set_margins(self, margins, title_height):
        self.margin_left, self.margin_top, self.margin_right, self.margin_bottom = map(int, margins)
        self.title_height = int(title_height)

    def set_gap(self):
        self.gap = random.randint(self.config.min_cell_gap, self.config.max_cell_gap) if self.config.enable_cell_gap else 0
        # 테이블 크기 계산 (갭 포함)
        self.table_width = min(max(self.config.min_table_width, 
                                    self.image_width - self.margin_left - self.margin_right), 
                               self.image_width)
        
        self.table_height = min(max(self.config.min_table_height, 
                                     self.image_height - self.margin_top - 
                                     self.margin_bottom - self.title_height), 
                                self.image_height)

    def set_matrix(self): 
        max_rows = self.config.max_rows
        max_cols = self.config.max_cols
    
        # 행과 열 수 계산 (갭 고려), 최소 2개의 셀 보장
        self.cols = max(2, min(max_cols, (self.table_width + self.gap) // (self.config.min_cell_width + self.gap)))
        self.rows = max(2, min(max_rows, (self.table_height + self.gap) // (self.config.min_cell_height + self.gap)))
        
        # 만약 행이나 열이 1인 경우, 다른 쪽을 2로 설정
        if self.rows == 1:
            self.cols = max(2, self.cols)
        elif self.cols == 1:
            self.rows = max(2, self.rows)

        
    def set_table_dimensions(self):
        """실제 테이블 크기를 설정합니다."""
        self.total_rows = random.randint(self.config.min_rows, self.config.max_rows)
        self.total_cols = random.randint(self.config.min_cols, self.config.max_cols)
        
        
    def set_cell_size_and_header(self):
        available_width = self.table_width - (self.cols - 1) * self.gap
        available_height = self.table_height - (self.rows - 1) * self.gap
        
        # 최소 셀 크기 계산
        min_cell_width = max(self.config.min_cell_width, available_width // self.cols)
        min_cell_height = max(self.config.min_cell_height, available_height // self.rows)

        # 헤더 행과 열의 크기 계산
        header_row_height = int(min_cell_height * self.config.header_row_height_factor)
        header_col_width = int(min_cell_width * self.config.header_col_width_factor)

        # 셀 크기 초기화
        self.col_widths = [min_cell_width] * self.cols
        self.row_heights = [min_cell_height] * self.rows


        # 헤더 행과 열에 대해 다른 크기 적용
        if self.config.table_type in ['header_row', 'header_both']:
            self.row_heights[0] = header_row_height
        if self.config.table_type in ['header_column', 'header_both']:
            self.col_widths[0] = header_col_width

        # 첫 번째 헤더 설정
        self.set_first_header()

    def set_first_header(self):
                # 0,0 셀(왼쪽 위 셀)의 너비만 랜덤하게 조정
        if self.config.table_type in ['header_column', 'header_both']:
            corner_cell_width_factor = random.uniform(0.5, 2.0)  # 50% ~ 200% 사이의 랜덤 값
            self.col_widths[0] = int(self.col_widths[0] * corner_cell_width_factor)

            # 너비가 최소값 이하로 내려가지 않도록 보정
            self.col_widths[0] = max(self.col_widths[0], self.config.min_cell_width)

            # 너비가 테이블 전체 너비의 50%를 넘지 않도록 제한
            max_width = int(self.table_width * 0.5)
            self.col_widths[0] = min(self.col_widths[0], max_width)

    def validate_cell_coordinates(self, cells, table_bbox):
        for cell in cells:
            # x 좌표 조정
            cell['x1'] = max(table_bbox[0], min(cell['x1'], table_bbox[2] - 1))
            cell['x2'] = min(table_bbox[2], max(cell['x2'], cell['x1'] + 1))
            
            # y 좌표 조정
            cell['y1'] = max(table_bbox[1], min(cell['y1'], table_bbox[3] - 1))
            cell['y2'] = min(table_bbox[3], max(cell['y2'], cell['y1'] + 1))
            
            # 최소 크기 보장
            if cell['x2'] <= cell['x1']:  # x2가 x1보다 작거나 같으면
                cell['x2'] = cell['x1'] + 1
            if cell['y2'] <= cell['y1']:  # y2가 y1보다 작거나 같으면
                cell['y2'] = cell['y1'] + 1
            
        return cells
    def process_remain_space(self):
    # 헤더 행과 열을 제외한 남은 공간 계산
        remaining_width = self.table_width - self.col_widths[0] - (self.cols - 1) * self.gap
        remaining_height = self.table_height - self.row_heights[0] - (self.rows - 1) * self.gap
        
        # 남은 열들의 너비 재조정
        if self.cols > 1:
            col_width = remaining_width // (self.cols - 1)
            extra_width = remaining_width % (self.cols - 1)
            for i in range(1, self.cols):
                self.col_widths[i] = col_width
                if extra_width > 0:
                    self.col_widths[i] += 1
                    extra_width -= 1
        
        # 남은 행들의 높이 재조정
        if self.rows > 1:
            row_height = remaining_height // (self.rows - 1)
            extra_height = remaining_height % (self.rows - 1)
            for i in range(1, self.rows):
                self.row_heights[i] = row_height
                if extra_height > 0:
                    self.row_heights[i] += 1
                    extra_height -= 1

        
    def make_cells(self):
        self.cells = []
        y = self.margin_top + self.title_height

        for row in range(self.rows):
            x = self.margin_left
            for col in range(self.cols):
                is_header = (self.config.table_type in ['header_row', 'header_both'] and row == 0) or \
                            (self.config.table_type in ['header_column', 'header_both'] and col == 0)
                is_gray = random.random() < self.config.gray_cell_probability

                cell = {
                    'x1': x,
                    'y1': y,
                    'x2': x + self.col_widths[col],
                    'y2': y + self.row_heights[row],
                    'row': row,
                    'col': col,
                    'is_merged': False,
                    'is_header': is_header,
                    'is_gray': is_gray,
                    'original_height': self.row_heights[row],
                    'overflow': None,
                    'cell_type': 'normal_cell'
                }
                self.cells.append(cell)
                x += self.col_widths[col] + self.gap
            y += self.row_heights[row] + self.gap

        # 테이블의 전체 경계 상자 계산
        table_bbox = [
            self.margin_left,
            self.margin_top + self.title_height,
            self.margin_left + self.table_width,
            self.margin_top + self.title_height + self.table_height
        ]
        
        if self.table_width > (table_bbox[2] - table_bbox[0]) or self.table_height > (table_bbox[3] - table_bbox[1]):
            table_logger.warning(f"Table size exceeds bounding box. Bbox: {table_bbox}, Table: {self.table_width}x{self.table_height}")
        
        # 셀 유효성 검사
        self.cells = validate_all_cells(self.cells, table_bbox, "cell creation")
        self.table_bbox = table_bbox
        return self.cells
    def make_merged_cells(self):
        
        if self.config.enable_cell_merging:
            self.cells = merge_cells(self.cells, self.rows, self.cols, self.config)
        # 유효성 검사
        self.cells = validate_all_cells(self.cells, self.table_bbox, "cell merging")
    def make_overflow_cells(self):
    # is_gray가 True인 셀만 오버플로우 처리
        gray_cells = [cell for cell in self.cells if cell.get('is_gray', False)]
        if gray_cells:
            if self.config.enable_overflow:
                overflowed_cells = plan_cell_overflow(self, gray_cells, self.config)
                # 오버플로우 처리된 셀을 기존 셀 리스트에 반영
                for overflowed_cell in overflowed_cells:
                    # 오버플로우된 셀을 cells에 추가하고 기존 gray 셀은 제거
                    if overflowed_cell not in self.cells:
                        self.cells.append(overflowed_cell)

            # 유효성 검사
                self.cells = validate_all_cells(self.cells, self.table_bbox, "cell merging")
    def validate_table(self):
        # 최종 검증: 셀이 2개 미만인 경우 처리
        if len(self.cells) < 2:
            table_logger.warning("셀 수가 2개 미만입니다. 테이블을 2x1로 재구성합니다.")
            self.cells = [
                {'x1': self.margin_left, 'y1': self.margin_top + self.title_height, 'x2': self.margin_left + self.table_width // 2,
                'y2': self.margin_top + self.title_height + self.table_height,
                'row': 0, 'col': 0, 'is_merged': False, 'is_header': False,
                'original_height': self.table_height, 'overflow': None, 'cell_type': 'normal_cell'},
                {'x1': self.margin_left + self.table_width // 2, 'y1': self.margin_top + self.title_height,
                'x2': self.margin_left + self.table_width,
                'y2': self.margin_top + self.title_height + self.table_height,
                'row': 0, 'col': 1, 'is_merged': False, 'is_header': False,
                'original_height': self.table_height, 'overflow': None, 'cell_type': 'normal_cell'}
            ]
        # 좌표 검증 호출
        self.cells = self.validate_cell_coordinates(self.cells, self.table_bbox)
        
    def create_table(self, image_width: int, image_height: int, margins, title_height, config: TableGenerationConfig):
        # 기존 구조 유지
        self.config = config
        self.config.randomize_settings()
        self.config.overflow_or_merged()
        
        # 테이블 크기 설정
        self.set_table_dimensions()

        # 이미지 크기 설정 및 마진 설정
        self.image_width = image_width
        self.image_height = image_height
        self.set_margins(margins, title_height)
        
        # 갭 및 행렬 설정
        self.set_gap()
        self.set_matrix()
        
        # 셀 크기 및 헤더 설정
        self.set_cell_size_and_header()
        
        # 남은 공간 처리 및 셀 생성
        self.process_remain_space()
        
        # 셀 생성 및 유효성 검사
        try:
            # 셀 생성 후 병합 및 오버플로우 처리 
            self.make_cells()
            self.make_merged_cells()
            self.make_overflow_cells()

            # 위치 조정 및 테이블 검증 
            adjust_cell_positions(self.cells, config, self.table_bbox)
            self.validate_table()

            is_table_rounded = config.enable_rounded_table_corners and random.random() < config.rounded_table_corner_probability
            table_corner_radius = random.randint(config.min_table_corner_radius, config.max_table_corner_radius) if is_table_rounded else 0
            
            return (self, 
                    [cell for cell in self.cells], 
                    self.table_bbox, 
                    is_table_rounded, 
                    table_corner_radius)

        except Exception as e:
            table_logger.error(f"테이블 생성 중 오류 발생: {str(e)}", exc_info=True)
            raise
    

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

        # 병합 여부와 관계없이 모든 셀의 y1, y2를 고려
        y1 = min(cell['y1'] for cell in row_cells)
        y2 = max(cell['y1'] + cell.get('original_height', cell['y2'] - cell['y1']) for cell in row_cells)

        row_coords = [table_bbox[0], y1, table_bbox[2], y2]
        is_header = i == 0 and config.table_type in ['header_row', 'header_both']

        # 헤더 행 어노테이션 생성
        category_id = HEADER_ROW_CATEGORY_ID if is_header else ROW_CATEGORY_ID
        row_annotation = create_annotation(row_coords, category_id, "row", {
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
        
        # 헤더 열 어노테이션 생성
        category_id = HEADER_COLUMN_CATEGORY_ID if is_header else COLUMN_CATEGORY_ID
        col_annotation = create_annotation(col_coords, category_id, "column", {
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
        overflow_applied = cell_info.get('overflow_applied', False)
        
        if overflow_applied:
            y1 = min(y1, cell_info.get('overflow_y1', y1))
            y2 = max(y2, cell_info.get('overflow_y2', y2))

        coords = [x1, y1, x2, y2]
        
        is_affected = (cell_info['row'], cell_info['col']) in affected_cells
        
        attributes = {
            "is_header": cell_info.get('is_header', False),
            "is_merged": cell_info.get('is_merged', False),
            "has_overflow": overflow_applied,
            "overflow_direction": overflow.get('direction', 'none') if overflow_applied else 'none',
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
            "y2": y2,
            "overflow_applied": overflow_applied
        }
        
        # 셀 카테고리 결정
        if cell_info.get('is_merged', False) and not overflow_applied:
            category_id = MERGED_CELL_CATEGORY_ID
            category_name = "merged_cell"
        elif overflow_applied:
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
    
    table_logger.info(f"generate_coco_annotations 종료: 이미지 ID {image_id}, 생성된 어노테이션 수 {len(coco_annotations)}")
    table_logger.info(f"테이블 구조: {len(cell_rows)}행 x {len(cell_cols)}열")
    table_logger.info(f"원래 의도된 구조: {config.total_rows}행 x {config.total_cols}열")
    table_logger.info(f"행 불일치: {abs(len(cell_rows) - config.total_rows)}, 열 불일치: {abs(len(cell_cols) - config.total_cols)}")

    return coco_annotations 

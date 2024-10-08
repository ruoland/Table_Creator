import random
from typing import Dict

CELL_CATEGORY_ID = 0

TABLE_CATEGORY_ID = 1
ROW_CATEGORY_ID = 2
COLUMN_CATEGORY_ID = 3
MERGED_CELL_CATEGORY_ID = 4
OVERFLOW_CELL_CATEGORY_ID = 5
MERGED_OVERFLOW_CELL_CATEGORY_ID = 6

MIN_FONT_SIZE = 5
MIN_CELL_SIZE_FOR_TEXT = 20
PADDING = 2
MIN_CELL_SIZE_FOR_CONTENT = 5
# 클래스 비율 설정 추가

class TableGenerationConfig:
    simple_table_ratio: float = 0.6
    medium_table_ratio: float = 0.3
    complex_table_ratio: float = 0.1
    dataset_counter = None

    max_rows_simple: int = 5
    max_cols_simple: int = 5
    
    max_rows_medium: int = 8
    max_cols_medium: int = 8
    
    max_rows_complex: int = 10
    max_cols_complex: int = 10
    cell_type_ratios = {
        'normal_cell': 0.70,
        'merged_cell': 0.09,
        'overflow_cell': 0.09,
        'merged_overflow_cell': 0.12
    }

    line_colors = {
        'black': (0, 0, 0),
        'dark_gray': (64, 64, 64),
        'gray': (128, 128, 128),
        'light_gray': (192, 192, 192),
        'blue': (0, 0, 255),
        'red': (255, 0, 0),
    }
    line_color_weights = {
        'black': 0.6,
        'dark_gray': 0.2,
        'gray': 0.1,
        'light_gray': 0.05,
        'blue': 0.03,
        'red': 0.02,
    }

    # 선 스타일 설정
    line_styles = ['solid', 'dashed', 'dotted']
    line_style_weights = {
        'solid': 0.7,
        'dashed': 0.2,
        'dotted': 0.1,
    }
    
    min_line_thickness: int = 1
    max_line_thickness: int = 5
    line_thickness_distribution: Dict[str, float] = {
        'thin': 0.4,
        'normal': 0.4,
        'thick': 0.2
    }

    def get_random_line_thickness(self):
        thickness_type = random.choices(list(self.line_thickness_distribution.keys()),
                                        weights=list(self.line_thickness_distribution.values()))[0]
        if thickness_type == 'thin':
            return max(1, random.randint(self.min_line_thickness, max(2, self.min_line_thickness + 1)))
        elif thickness_type == 'normal':
            return max(1, random.randint(min(3, self.max_line_thickness - 1), max(3, self.max_line_thickness - 1)))
        else:  # thick
            return max(1, random.randint(min(4, self.max_line_thickness), self.max_line_thickness))
    def __init__(self):
        
        table_type_weights = [0.2, 0.3, 0.3, 0.2]  # 각 유형의 확률 조정
        table_types = ['no_header', 'header_row', 'header_column', 'header_both']

        self.table_type = random.choices(table_types, weights=table_type_weights)[0]

        self.config_mode = 'Random' # None 혹은 Random
        # 이미지 크기 및 해상도 설정
        self.total_images = 10000 # 생성할 총 이미지 수
        self.min_image_width, self.max_image_width = 800, 2400
        self.min_image_height, self.max_image_height = 600, 2400
        # 테이블 구조 설정
        self.min_cols, self.max_cols = 4, 45
        self.min_rows, self.max_rows = 4, 45
        self.min_table_width, self.min_table_height = 200, 100
        self.enable_colored_cells = True
        # 실제 테이블 크기 (초기값은 None으로 설정)
        self.enable_divider_lines = True
        self.horizontal_divider_probability = 0.2
        self.vertical_divider_probability = 0.2
        self.divider_line_thickness_range = (2, 5)  # 구분선의 두께 범위
        self.total_rows = None # 알아서 
        self.total_cols = None # 알아서 적용되는 설정
        self.line_break_probability = 0.1
        self.line_thickness_variation_probability = 0.1
        self.corner_imperfection_probability = 0.1
        self.texture_effect_probability = 0.1
        self.line_blur_probability = 0.2
        # 셀 병합 및 오버플로우 설정 조정
        # 셀 설정
        self.min_cell_width, self.max_cell_width = 5, 200
        self.min_cell_height, self.max_cell_height = 5, 300
        self.enable_cell_gap = True  # 추가된 부분
        self.min_cell_gap, self.max_cell_gap = 0, 3
        self.cell_no_border_probability = 0.2  # 셀에 테두리를 그리지 않을 확률
        enable_colored_cells: bool = True
        # 둥근 셀 설정
        self.enable_rounded_corners = True
        self.rounded_corner_probability = 0.3
        self.min_corner_radius = 2
        self.max_corner_radius = 8
        self.background_shape_count = 20  # 배경에 추가할 도형의 수
        # 헤더 설정
        self.header_row_height_factor = 1.0
        self.header_col_width_factor = 1.0
        self.no_side_borders_cells_probability = 0.4  # 양옆 선 없는 셀의 확률

        self.no_side_borders_row_probability = 0.3  # 양옆 선이 없는 행의 확률
        self.background_shape_count = 20  # 배경에 추가할 도형의 수

        # 여백 및 테두리 설정
        self.min_margin, self.max_margin = 10, 100
        self.enable_outer_border = True
        self.outer_border_probability = 0.8
        
        self.min_line_thickness = max(1, random.randint(1, 3))
        self.max_line_thickness = max(1, random.randint(self.min_line_thickness + 2, 5))
        self.min_outer_line_thickness, self.max_outer_line_thickness = 1, 4
        self.no_side_borders_cells_probability = 0.4  # 양옆 선 없는 셀의 확률
        
        
        self.enable_rounded_table_corners: bool = True
        self.rounded_table_corner_probability: float = 0.7
        self.min_table_corner_radius: int = 5
        self.max_table_corner_radius: int = 30
        # 제목 설정
        self.enable_title = True
        self.min_title_size, self.max_title_size = 20, 40

        # 셀 내용 설정
        self.cell_content_types = ['text', 'shapes', 'mixed']
        self.text_positions = ['center', 'top_left', 'top_right', 'bottom_left', 'bottom_right']
        self.enable_text_generation = True
        self.max_text_length = 50
        self.class_info_probability = 0.4
        self.common_word_probability = 0.5
        # 오버플로우 설정
        self.enable_overflow = True
        self.overflow_probability = 0.4
        self.max_overflow_height = 50
        self.min_overflow_height = 10  # 최소 오버플로우 높이
        self.irregular_thickness_probability = 0.3
        self.line_curve_probability = 0.3
        self.color_variation_probability = 0.2
        self.end_imperfection_probability = 0.3
        self.transparency_variation_probability = 0.2
  
        
        # 도형 설정
        self.enable_shapes = True
        self.shape_types = ['rectangle', 'circle', 'triangle', 'line', 'arc', 'polygon']
        self.min_shape_size, self.max_shape_size = 5, 50
        self.min_shapes, self.max_shapes = 1, 5
        self.shape_line_width = 2
        self.shape_generation_ratio = 0.2
        # 그룹 헤더 관련 설정
        self.enable_group_headers = False
        self.group_header_probability = 0.5
        self.max_group_header_rows = 2
        self.max_group_header_cols = 3
        self.group_header_offset = 10  # 그룹 헤더의 오프셋 (픽셀 단위)
        self.group_header_min_height = 20  # 그룹 헤더의 최소 높이
        self.group_header_min_width = 20   # 그룹 헤더의 최소 너비
        self.group_header_height = 30  # 그룹 헤더의 높이
        self.min_rows_for_group_header = 8  # 그룹 헤더를 추가하기 위한 최소 행 수
        self.group_header_interval = random.randint(3, 5)  # 그룹 헤더 간격
        self.min_cell_width = 5   # 일반 셀의 최소 너비
        self.min_cell_height = 5  # 일반 셀의 최소 높이
        # 셀 병합 설정
        self.enable_cell_merging = True
        self.merged_cell_probability = random.uniform(0.3, 0.6)
        self.merged_cell_ratio = random.uniform(0.4, 0.6)
        self.max_horizontal_merge = random.randint(2, 5)
        self.max_vertical_merge = random.randint(2, 5)

        self.horizontal_merge_probability = random.uniform(0.1, 0.3)
        self.vertical_merge_probability = random.uniform(0.1, 0.4)
        self.no_border_cell_probability = 0.1  # 테두리가 없는 셀의 확률
        # 색상 설정
        self.background_colors = {
            'light': {'white': (255, 255, 255)},
            'dark': {'black': (0, 0, 0)}
        }
        self.color_brightness_threshold = 128
        self.dark_gray_range = (0, 100)
        self.light_gray_range = (165, 255)
        self.medium_gray_range = (50, 150)
        self.light_medium_gray_range = (110,210)
        self.faded_color_probability = 0.4

        # 불완전한 선 그리기 설정
        self.enable_imperfect_lines = False
        self.imperfect_line_probability = {
            'top': 0.9,    # 90% 확률로 상단 선 그리기
            'bottom': 0.9, # 90% 확률로 하단 선 그리기
            'left': 0.95,  # 95% 확률로 좌측 선 그리기
            'right': 0.95  # 95% 확률로 우측 선 그리기
        }
        # 회색 셀 설정
        self.enable_gray_cells = False
        self.gray_cell_probability = 0.0
        self.gray_color_range = (50, 220)
        self.no_gray_cell_probability = 0.0
        self.gray_cell_no_border_probability: float = 0.0
        # 스타일 설정
        self.styles = ['thin', 'medium', 'thick', 'double']
        self.fonts = ['fonts/NanumGothic.ttf', 'fonts/SANGJU Dajungdagam.ttf', 'fonts/SOYO Maple Regular.ttf']
        self.line_styles = ['solid', 'dashed', 'dotted']

        # 불완전성 및 효과 설정
        self.enable_table_imperfections = True
        self.line_imperfection_probability = 0.3 # 돌출부나 선 주변에 점 추가할 확률
        self.imperfect_ratio = 0.3
        self.enable_cell_inside_imperfections = True
        self.enable_random_lines = True
        self.random_line_probability = 0.3
        self.cell_imperfection_probability = 0.3

        # 특수 효과 설정
        self.enable_noise = True
        self.noise_intensity_range = (0.01, 0.05)
        self.enable_blur = True
        self.blur_radius_range = (0.3, 1.0)
        self.blur_probability = 0.3
        self.enable_brightness_variation = True
        self.enable_contrast_variation = True
        self.brightness_factor_range = (0.7, 1.3)
        self.contrast_factor_range = (0.7, 1.3)
        self.enable_perspective_transform = False
        self.perspective_transform_range = (0.05, 0.17)
        self.perspective_intensity = 0.1
        self.perspective_direction = None
        self.enable_shadow = True
        self.shadow_opacity_range = (30, 150)
        self.shadow_blur_radius = 5
        self.shadow_size_ratio = 0.1
        self.shadow_gradient_strength = 1.0



        # 선 색상 설정
        
        # 기타 설정
        self.empty_cell_ratio = 0.1
        self.enable_background_shapes = False

        
    def set_table_dimensions(self):
        """실제 테이블 크기를 설정합니다."""
        self.total_rows = random.randint(self.min_rows, self.max_rows)
        self.total_cols = random.randint(self.min_cols, self.max_cols)

    def disable_all_effects(self):
        # 셀 및 테이블 구조 관련
        self.enable_cell_gap = False
        self.enable_rounded_corners = False
        self.enable_outer_border = False
        self.enable_cell_merging = False
        self.enable_horizontal_merge = False
        self.enable_vertical_merge = False
        self.enable_rounded_table_corners = False
        self.enable_divider_lines = False
        self.enable_group_headers = False

        # 내용 관련
        self.enable_title = False
        self.enable_text_generation = False
        self.enable_text_wrapping = False
        self.enable_shapes = False
        self.enable_overflow = False

        # 색상 관련
        self.enable_gray_cells = False
        self.enable_colored_cells = False

        # 불완전성 및 효과 관련
        self.enable_table_imperfections = False
        self.enable_random_lines = False
        self.enable_cell_inside_imperfections = False
        self.enable_noise = False
        self.enable_blur = False
        self.enable_brightness_variation = False
        self.enable_contrast_variation = False
        self.enable_perspective_transform = False
        self.enable_shadow = False
        self.enable_table_cropping = False
        self.enable_background_shapes = False
        self.enable_imperfect_lines = False

        # 확률 관련 설정을 0으로 설정
        self.cell_no_border_probability = 0.0
        self.rounded_corner_probability = 0.0
        self.outer_border_probability = 0.0
        self.overflow_probability = 0.0
        self.gray_cell_probability = 0.0
        self.line_imperfection_probability = 0.0
        self.random_line_probability = 0.0
        self.cell_imperfection_probability = 0.0
        self.blur_probability = 0.0
        self.table_crop_probability = 0.0
        self.empty_cell_ratio = 0.0
        self.merged_cell_probability = 0.0
        self.horizontal_merge_probability = 0.0
        self.vertical_merge_probability = 0.0
        self.table_side_line_probability = 0.0
        self.rounded_table_corner_probability = 0.0
        self.horizontal_divider_probability = 0.0
        self.vertical_divider_probability = 0.0
        self.no_border_cell_probability = 0.0
        self.gray_cell_no_border_probability = 0.0
        self.faded_color_probability = 0.0
        self.line_break_probability = 0.0
        self.line_thickness_variation_probability = 0.0
        self.corner_imperfection_probability = 0.0
        self.texture_effect_probability = 0.0
        self.line_blur_probability = 0.0
        self.irregular_thickness_probability = 0.0
        self.line_curve_probability = 0.0
        self.color_variation_probability = 0.0
        self.end_imperfection_probability = 0.0
        self.transparency_variation_probability = 0.0

        # 선 관련 설정
        self.line_styles = ['solid']
        self.line_color = (0, 0, 0)  # 검은색으로 고정
        self.min_line_thickness = 1
        self.max_line_thickness = 1

        # 기타 설정
        self.imperfect_ratio = 0.0
        self.class_info_probability = 0.0
        self.common_word_probability = 0.0

        # 헤더 관련 설정
        self.header_row_height_factor = 1.0
        self.header_col_width_factor = 1.0

        # 폰트 및 텍스트 관련 설정
        self.fonts = ['fonts/NanumGothic.ttf']  # 기본 폰트만 사용
        self.min_font_size = 10
        self.max_font_size = 10

        # 크기 및 범위 설정
        self.min_corner_radius = 0
        self.max_corner_radius = 0
        self.min_table_corner_radius = 0
        self.max_table_corner_radius = 0
        self.noise_intensity_range = (0, 0)
        self.blur_radius_range = (0, 0)
        self.brightness_factor_range = (1, 1)
        self.contrast_factor_range = (1, 1)
        self.shadow_opacity_range = (0, 0)
        self.shadow_blur_radius = 0
        self.shadow_size_ratio = 0
        self.max_crop_ratio = 0


    def test_disable_all_effects(self):
        self.disable_all_effects()
        
        # 모든 enable_ 설정이 False인지 확인
        for attr_name in dir(self):
            if attr_name.startswith('enable_'):
                assert getattr(self, attr_name) == False, f"{attr_name} should be False"
        
        # 모든 확률 관련 설정이 0인지 확인
        probability_attrs = [
            'cell_no_border_probability', 'rounded_corner_probability', 'outer_border_probability',
            'overflow_probability', 'gray_cell_probability', 'line_imperfection_probability',
            'random_line_probability', 'cell_imperfection_probability', 'blur_probability',
            'table_crop_probability', 'empty_cell_ratio', 'merged_cell_probability',
            'horizontal_merge_probability', 'vertical_merge_probability', 'table_side_line_probability',
            'rounded_table_corner_probability', 'horizontal_divider_probability',
            'vertical_divider_probability', 'no_border_cell_probability',
            'gray_cell_no_border_probability', 'faded_color_probability', 'line_break_probability',
            'line_thickness_variation_probability', 'corner_imperfection_probability',
            'texture_effect_probability', 'line_blur_probability', 'irregular_thickness_probability',
            'line_curve_probability', 'color_variation_probability', 'end_imperfection_probability',
            'transparency_variation_probability', 'imperfect_ratio', 'class_info_probability',
            'common_word_probability'
        ]
        for attr_name in probability_attrs:
            assert getattr(self, attr_name) == 0.0, f"{attr_name} should be 0.0"
        
        # 기타 설정 확인
        assert self.line_styles == ['solid'], "line_styles should be ['solid']"
        assert self.line_color == (0, 0, 0), "line_color should be (0, 0, 0)"
        assert self.min_line_thickness == 1, "min_line_thickness should be 1"
        assert self.max_line_thickness == 1, "max_line_thickness should be 1"
        assert self.fonts == ['fonts/NanumGothic.ttf'], "fonts should be ['fonts/NanumGothic.ttf']"
        assert self.min_font_size == 10, "min_font_size should be 10"
        assert self.max_font_size == 10, "max_font_size should be 10"
        assert self.header_row_height_factor == 1.0, "header_row_height_factor should be 1.0"
        assert self.header_col_width_factor == 1.0, "header_col_width_factor should be 1.0"
        
        # 범위 설정 확인
        assert self.min_corner_radius == 0, "min_corner_radius should be 0"
        assert self.max_corner_radius == 0, "max_corner_radius should be 0"
        assert self.min_table_corner_radius == 0, "min_table_corner_radius should be 0"
        assert self.max_table_corner_radius == 0, "max_table_corner_radius should be 0"
        assert self.noise_intensity_range == (0, 0), "noise_intensity_range should be (0, 0)"
        assert self.blur_radius_range == (0, 0), "blur_radius_range should be (0, 0)"
        assert self.brightness_factor_range == (1, 1), "brightness_factor_range should be (1, 1)"
        assert self.contrast_factor_range == (1, 1), "contrast_factor_range should be (1, 1)"
        assert self.shadow_opacity_range == (0, 0), "shadow_opacity_range should be (0, 0)"
        assert self.shadow_blur_radius == 0, "shadow_blur_radius should be 0"
        assert self.shadow_size_ratio == 0, "shadow_size_ratio should be 0"
        assert self.max_crop_ratio == 0, "max_crop_ratio should be 0"


    def apply_simple_config(self):
        self.disable_all_effects()
        self.enable_simple_settings()
        self.randomize_simple_settings()
    def randomize_settings(self):
        # 셀 타입 비율 랜덤화 (옵션)
   
        # 이미지 크기 랜덤화
        self.min_image_width = random.randint(800, 3000)
        self.max_image_width = random.randint(self.min_image_width, 4000)
        self.min_image_height = random.randint(600, 2000)
        self.max_image_height = random.randint(self.min_image_height, 3000)

        self.line_break_probability = random.uniform(0.1, 0.3)
        self.line_thickness_variation_probability = random.uniform(0.1, 0.3)
        self.corner_imperfection_probability = random.uniform(0.1, 0.3)
        self.texture_effect_probability = random.uniform(0.1, 0.3)
        self.line_blur_probability = random.uniform(0.1, 0.3)
        # 불완전한 선 그리기 설정 랜덤화
        self.enable_imperfect_lines = random.choices([True, False], weights=[5, 5])[0]
        if self.enable_imperfect_lines:
            self.imperfect_line_probability = {
                'top': random.uniform(0.2, 0.4),
                'bottom': random.uniform(0.2, 0.4),
                'left': random.uniform(0.4, 0.6),
                'right': random.uniform(0.4, 0.6)
            }
        self.no_side_borders_probability = random.uniform(0.2, 0.4)
        self.min_cell_width = random.randint(20, 100)
        self.max_cell_width = random.randint(self.min_cell_width, 300)
        self.min_cell_height = random.randint(20, 80)
        self.max_cell_height = random.randint(self.min_cell_height, 200)

       # 셀 설정 랜덤화
        self.min_cell_gap = random.randint(0, 2)
        self.max_cell_gap = random.randint(self.min_cell_gap, 5)
        self.enable_cell_gap = random.choice([True, False])  # 추가된 부분
        
        # 테이블 구조 랜덤화
        self.min_table_width = random.randint(300, 800)
        self.min_table_height = random.randint(300, 800)
        self.min_cols = random.randint(2, 15)
        self.max_cols = random.randint(self.min_cols, 30)
        self.min_rows = random.randint(2, 30)
        self.max_rows = random.randint(self.min_rows, 50)
                # 테이블 크기 재설정
        self.set_table_dimensions()
 
        # 둥근 모서리 설정 랜덤화
        self.enable_rounded_corners = random.choice([True, False])
        if self.enable_rounded_corners:
            self.rounded_corner_probability = random.uniform(0.3, 0.6)
            self.min_corner_radius = random.randint(5, 10)
            self.max_corner_radius = random.randint(self.min_corner_radius, 40)
            
        self.table_side_line_probability = random.uniform(0.2, 0.5)
                
        # 헤더 설정 랜덤화
        self.header_row_height_factor = random.uniform(0.2, 2)
        self.header_col_width_factor = random.uniform(0.2, 1.5)
        
        #선 설정
        self.irregular_thickness_probability = random.uniform(0.2, 0.4)
        self.line_curve_probability = random.uniform(0.2, 0.4)
        self.color_variation_probability = random.uniform(0.1, 0.3)
        self.end_imperfection_probability = random.uniform(0.2, 0.5)
        self.transparency_variation_probability = random.uniform(0.1, 0.3)
        
        
        # 굵은 선
        self.horizontal_divider_probability = random.uniform(0.2, 0.4)
        self.vertical_divider_probability = random.uniform(0.2, 0.4)
        self.enable_divider_lines = random.choice([True, False])
        
        # 여백 및 테두리 설정 랜덤화
        self.min_margin = random.randint(0, 200)  # 최소 여백을 0부터 시작
        self.max_margin = random.randint(self.min_margin, 700)  # 최대 여백을 크게 설정
        self.enable_outer_border = random.choice([True, False])
        self.outer_border_probability = random.uniform(0.8, 0.9)

        self.min_outer_line_thickness = random.randint(1, 2)
        self.max_outer_line_thickness = random.randint(self.min_outer_line_thickness + 1, 6)

        # 제목 설정 랜덤화
        self.enable_title = random.choice([True, False])
        self.min_title_size = random.randint(15, 25)
        self.max_title_size = random.randint(self.min_title_size, 50)
        self.no_border_cell_probability = random.uniform(0.3, 0.6)
        
        # 셀 내용 설정 랜덤화
        self.enable_text_generation = random.choice([True, False])
        self.enable_text_wrapping = random.choice([True, False])
        self.max_text_length = random.randint(30, 100)
        self.class_info_probability = random.uniform(0.2, 0.6)
        self.common_word_probability = random.uniform(0.3, 0.7)
        
        self.enable_overflow = random.choices([True, False], weights=[5, 5])[0]  # 항상 활성화
        self.overflow_probability = random.uniform(0.3, 0.6)
        self.max_overflow_height = random.randint(15, 60)

        # 도형 설정 랜덤화
        self.enable_shapes = random.choice([True, False])
        self.min_shapes = random.randint(1, 4)
        self.max_shapes = random.randint(self.min_shapes, 10)
        self.shape_line_width = random.randint(1, 4)
        self.shape_generation_ratio = random.uniform(0.1, 0.3)


        # 셀 병합 설정 랜덤화
        self.enable_cell_merging = random.choices([True, False], weights=[6, 4])[0]  # 항상 활성화
        self.enable_horizontal_merge = random.choices([True, False], weights=[6, 4])[0] # 항상 활성화
        self.enable_vertical_merge = random.choices([True, False], weights=[6, 4])[0]   # 항상 활성화
        self.merged_cell_ratio = random.uniform(0.4, 0.6)
        self.merged_cell_probability = random.uniform(0.4, 0.6)
        self.max_horizontal_merge = random.randint(2, 4)
        self.max_vertical_merge = random.randint(2, 4)
        self.horizontal_merge_probability = random.uniform(0.3, 0.6)
        self.vertical_merge_probability = random.uniform(0.3, 0.6)
        # 색상 설정 랜덤화
        self.color_brightness_threshold = random.randint(100, 150)
        self.dark_gray_range = (random.randint(0, 32), random.randint(33, 96))
        self.light_gray_range = (random.randint(160, 224), 255)
        self.medium_gray_range = (random.randint(32, 96), random.randint(97, 160))
        self.light_medium_gray_range = (random.randint(96, 160), random.randint(161, 224))
        self.faded_color_probability = random.uniform(0.1, 0.4)
        
        # 컬러 셀 설정 랜덤화 (True: 40%, False: 50%)
        self.enable_colored_cells = random.choices([True, False], weights=[4, 6])[0]
        self.enable_gray_cells = random.choices([True, False], weights=[4, 6])[0]

        # 회색 셀 설정 랜덤화 (True: 40%, False: 70%)
        if self.enable_gray_cells:
            self.gray_cell_probability = random.uniform(0.2, 0.6)
            min_gray = random.randint(30, 100)
            max_gray = random.randint(min_gray + 50, 240)
            self.gray_color_range = (min_gray, max_gray)
        self.no_gray_cell_probability = random.uniform(0.2, 0.3)

        # 불완전성 및 효과 설정 랜덤화
        self.enable_table_imperfections = random.choices([True, False], weights=[4, 6])[0]
        self.line_imperfection_probability = random.uniform(0.4, 0.6)
        self.imperfect_ratio = random.uniform(0.4, 0.7)
        self.enable_random_lines = random.choices([True, False], weights=[6, 4])[0]
        self.enable_cell_inside_imperfections = random.choices([True, False], weights=[6, 4])[0]
        self.random_line_probability = random.uniform(0.3, 0.6)
        self.cell_imperfection_probability = random.uniform(0.1, 0.4)

        # 특수 효과 설정 랜덤화

        #노이즈
        self.enable_noise = random.choices([True, False], weights=[7, 3])[0]
        if self.enable_noise:
            self.noise_intensity_range = (random.uniform(0.01, 0.05), random.uniform(0.05, 0.1))
        #블러
        self.enable_blur = random.choices([True, False], weights=[7, 3])[0]
        if self.enable_blur:
            self.blur_radius_range = (random.uniform(0.3, 0.7), random.uniform(0.7, 1.5))
            self.blur_probability = random.uniform(0.1, 0.5)
        #밝기/대비
        self.enable_brightness_variation = random.choices([True, False], weights=[7, 3])[0]
        self.enable_contrast_variation = random.choices([True, False], weights=[7, 3])[0]
        
        if self.enable_brightness_variation:
            self.brightness_factor_range = (random.uniform(0.7, 1.0), random.uniform(1.0, 1.3))
        if self.enable_contrast_variation:
            self.contrast_factor_range = (random.uniform(0.7, 1.0), random.uniform(1.0, 1.3))
        
        #그림자
        self.enable_shadow = random.choices([True, False], weights=[6, 4])[0]
        if self.enable_shadow:
            self.shadow_opacity_range = (random.randint(30, 70), random.randint(70, 150))
            self.shadow_blur_radius = random.uniform(1, 25)
            self.shadow_size_ratio = random.uniform(0.05, 0.25)

        # 테이블 잘림 설정 랜덤화
        self.enable_table_cropping = random.choices([True, False], weights=[3, 7])[0]
        if self.enable_table_cropping:
            self.table_crop_probability = random.uniform(0.1, 0.2)
            self.max_crop_ratio = random.uniform(0.2, 0.5)

            

        # 기타 설정 랜덤화
        self.empty_cell_ratio = random.uniform(0.1, 0.2)
        self.enable_background_shapes = random.choice([True, False])
        
    from typing import Tuple
    def get_random_pastel_color(self, bg_color: Tuple[int, int, int]):
        if not self.enable_colored_cells:
            return None
        
        # 배경 밝기 계산
        bg_brightness = sum(bg_color) / 3
        
        # 밝은 배경용 파스텔 색상
        light_pastel_colors = [
            (255, 255, 255),  # 흰색
            (255, 182, 193),  # 연한 빨간색 (분홍색)
            (173, 216, 230),  # 연한 파란색 (하늘색)
            (144, 238, 144),  # 연한 초록색
            (255, 255, 224),  # 연한 노란색
            (230, 230, 250),  # 연한 보라색
            (255, 218, 185),  # 연한 주황색
            (211, 211, 211),  # 연한 회색
            (175, 238, 238),  # 민트색
        ]
        
        # 어두운 배경용 색상
        dark_pastel_colors = [
            (100, 149, 237),  # 콘플라워 블루
            (147, 112, 219),  # 미디엄 퍼플
            (60, 179, 113),   # 미디엄 시 그린
            (255, 165, 0),    # 오렌지
            (255, 99, 71),    # 토마토
            (70, 130, 180),   # 스틸 블루
            (186, 85, 211),   # 미디엄 오키드
            (105, 105, 105),  # 딤 그레이
        ]
        
        # 배경 밝기에 따라 색상 리스트 선택
        if bg_brightness > 127:
            color_list = light_pastel_colors
        else:
            color_list = dark_pastel_colors
        
        # 랜덤하게 색상 선택
        base_color = random.choice(color_list)
        
        # 약간의 변화를 주기 위해 각 채널에 -10 ~ +10 사이의 랜덤한 값을 더함
        variation = lambda x: max(0, min(255, x + random.randint(-10, 10)))
        return tuple(variation(c) for c in base_color)
    def get_group_header_color(self, bg_color):
            # 배경색과 구분되는 그룹 헤더 색상 생성
            r, g, b = bg_color
            offset = 30  # 색상 차이
            return (
                max(0, min(255, r + random.randint(-offset, offset))),
                max(0, min(255, g + random.randint(-offset, offset))),
                max(0, min(255, b + random.randint(-offset, offset)))
            )

    def get_group_header_line_thickness(self):
        # 그룹 헤더의 선 두께 결정
        return random.randint(2, 4)

    def get_random_gray_color(self):
        if not self.enable_gray_cells:
            return None
        gray_value = random.randint(*self.gray_color_range)
        return (gray_value, gray_value, gray_value)


class DatasetCellTypeCounter:
    def __init__(self, config):
        self.total_images = config.total_images
        self.cell_type_ratios = config.cell_type_ratios
        self.current_counts = {t: 0 for t in self.cell_type_ratios}
        self.total_cells = 0
        self.images_created = 0

    def get_target_ratios(self):
        remaining_cells = max(0, int(self.total_cells * (self.total_images - self.images_created) / self.total_images))
        target_counts = {t: int(r * remaining_cells) for t, r in self.cell_type_ratios.items()}
        
        for cell_type, count in self.current_counts.items():
            target_counts[cell_type] = max(0, target_counts[cell_type] - count)
        
        total_remaining = sum(target_counts.values())
        if total_remaining == 0:
            return self.cell_type_ratios

        return {t: count / total_remaining for t, count in target_counts.items()}

    def update_counts(self, new_cells):
        for cell in new_cells:
            cell_type = cell.get('cell_type', 'normal_cell')  # 기본값 설정
            self.current_counts[cell_type] = self.current_counts.get(cell_type, 0) + 1
        self.total_cells += len(new_cells)
        self.images_created += 1
config = TableGenerationConfig()


import random
from typing import Dict
from typing import Tuple
import colorsys
import random
CELL_CATEGORY_ID = 0
TABLE_CATEGORY_ID = 1
ROW_CATEGORY_ID = 2
COLUMN_CATEGORY_ID = 3
MERGED_CELL_CATEGORY_ID = 4
OVERFLOW_CELL_CATEGORY_ID = 5
HEADER_ROW_CATEGORY_ID = 6
HEADER_COLUMN_CATEGORY_ID = 7


MIN_FONT_SIZE = 8
MIN_CELL_SIZE_FOR_TEXT = 20
PADDING = 2
MIN_CELL_SIZE_FOR_CONTENT = 5
# 클래스 비율 설정 추가

class TableGenerationConfig:
    simple_table_ratio: float = 0.6
    medium_table_ratio: float = 0.3
    complex_table_ratio: float = 0.1

    
    cell_type_ratios = {
        'normal_cell': 0.70,
        'merged_cell': 0.09,
        'overflow_cell': 0.09
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

    def __init__(self):
        
        # 테이블 유형 및 기본 설정
        table_types = ['no_header', 'header_row', 'header_column', 'header_both']  # 가능한 테이블 유형들
        self.table_type = random.choice(table_types)  # 랜덤으로 테이블 유형 선택
        self.config_mode = 'Random'  # 설정 모드 (None 또는 Random)
        self.total_images = 100  # 생성할 총 이미지 수
        self.image_level = 5.6 # 1은 셀, 표, 행, 열만 탐지하고, 2는 헤더 행 열, 2.6는 적은 수의 병합 셀, 오버플로, 3.1부터는 병합된 셀과 오버플로우 약간씩, 4.1부터는 병합된 셀과 오버플로우 셀 많이, 5.1부터는 불완전한 표, 선 없는 셀, 색깔 셀에서 선 안 그리기, 등.
        # 이미지 크기 및 해상도 설정
        self.min_image_width, self.max_image_width = 800, 2400  # 이미지 너비 범위
        self.min_image_height, self.max_image_height = 600, 2400  # 이미지 높이 범위
        self.predefined_grays = [
            50,   # 매우 어두운 회색
            80,   # 어두운 회색
            120,  # 중간 회색
            160,  # 밝은 회색
            200   # 매우 밝은 회색
        ]
        self.gray_variation = 20  # 회색 값의 변동 범위
        # 테이블 구조 설정
        self.min_cols, self.max_cols = 4, 10  # 열 수 범위
        self.min_rows, self.max_rows = 4, 10  # 행 수 범위
        self.min_table_width, self.min_table_height = 200, 100  # 최소 테이블 크기
        self.total_rows = None  # 실제 행 수 (자동 설정)
        self.total_cols = None  # 실제 열 수 (자동 설정)

        # 셀 설정
        self.min_cell_width, self.max_cell_width = 5, 200  # 셀 너비 범위
        self.min_cell_height, self.max_cell_height = 5, 300  # 셀 높이 범위
        self.enable_cell_gap = True  # 셀 간격 사용 여부
        self.min_cell_gap, self.max_cell_gap = 0, 3  # 셀 간격 범위
        self.cell_no_border_probability = 0  # 셀에 테두리를 그리지 않을 확률
        
        
        # 테두리 및 선 설정
        self.enable_divider_lines = True  # 구분선 사용 여부
        self.horizontal_divider_probability = 0.2  # 수평 구분선 확률
        self.vertical_divider_probability = 0.2  # 수직 구분선 확률
        self.divider_line_thickness_range = (2, 5)  # 구분선 두께 범위
        self.enable_outer_border = True  # 외곽선 사용 여부
        self.outer_border_probability = 0.8  # 외곽선 그릴 확률
        self.min_line_thickness = max(1, random.randint(1, 3))  # 최소 선 두께
        self.max_line_thickness = max(1, random.randint(self.min_line_thickness + 2, 5))  # 최대 선 두께
        self.min_outer_line_thickness, self.max_outer_line_thickness = 1, 4  # 외곽선 두께 범위

        # 모서리 및 형태 설정
        self.enable_rounded_corners = True  # 둥근 모서리 사용 여부
        self.rounded_corner_probability = 0.5  # 둥근 모서리 확률
        self.min_corner_radius = 2  # 최소 모서리 반경
        self.max_corner_radius = 7  # 최대 모서리 반경
        self.enable_rounded_table_corners = True  # 테이블 모서리 둥글게 처리 여부
        self.rounded_table_corner_probability = 0.7  # 테이블 모서리 둥글게 처리 확률
        self.min_table_corner_radius = 5  # 최소 테이블 모서리 반경
        self.max_table_corner_radius = 24  # 최대 테이블 모서리 반경

        # 헤더 설정
        self.header_row_height_factor = 1.0  # 헤더 행 높이 배수
        self.header_col_width_factor = 1.0  # 헤더 열 너비 배수
        self.no_side_borders_cells_probability = 0.2 # 양옆 선 없는 셀의 확률

        # 여백 설정
        self.min_margin, self.max_margin = 10, 100  # 여백 범위

        # 제목 설정
        self.enable_title = True  # 제목 사용 여부
        self.min_title_size, self.max_title_size = 20, 40  # 제목 크기 범위

        # 셀 내용 설정
        self.cell_content_types = ['text', 'shapes', 'mixed']  # 셀 내용 유형
        self.text_positions = ['center', 'top_left', 'top_right', 'bottom_left', 'bottom_right']  # 텍스트 위치 옵션
        self.enable_text_generation = True  # 텍스트 생성 사용 여부
        self.max_text_length = 50  # 최대 텍스트 길이
        self.class_info_probability = 0  # 클래스 정보 포함 확률
        self.common_word_probability = 0.5  # 일반 단어 사용 확률
        self.empty_cell = False
        self.empty_cell_probability = 0.3

        # 오버플로우 설정
        self.enable_overflow = True  # 오버플로우 사용 여부
        self.overflow_probability = 0.7  # 오버플로우 발생 확률
        self.min_overflow_height = 3  # 최소 오버플로우 높이
        self.max_overflow_height = 10  # 오버플로우 영향을 확인할 최대 행 수
        self.overflow_only_row_probability = 0.1
        self.default_row_height = 30
        self.default_col_width = 100
        self.row_overflow_probability = 0.4
        # 도형 설정
        self.enable_shapes = True  # 도형 사용 여부
        self.shape_types = ['rectangle', 'circle', 'triangle', 'line', 'arc', 'polygon']  # 사용 가능한 도형 유형
        self.min_shape_size, self.max_shape_size = 5, 50  # 도형 크기 범위
        self.min_shapes, self.max_shapes = 1, 5  # 도형 개수 범위
        self.shape_line_width = 2  # 도형 선 두께

       
        # 셀 병합 설정
        self.enable_cell_merging = True  # 셀 병합 사용 여부
        self.merged_cell_probability = 0.5  # 셀 병합 확률
        self.merged_cell_ratio = 0.5  # 병합된 셀 비율
        self.max_horizontal_merge = 4  # 최대 수평 병합 수
        self.max_vertical_merge = 4  # 최대 수직 병합 수
        self.horizontal_merge_probability = 0.3  # 수평 병합 확률
        self.vertical_merge_probability = 0.3  # 수직 병합 확률

        # 색상 설정
        self.background_colors = {
            'light': {'white': (255, 255, 255)},
            'dark': {'black': (0, 0, 0)}
        }  # 배경색 옵션
        self.color_brightness_threshold = 128  # 색상 밝기 임계값
        self.dark_gray_range = (0, 100)  # 어두운 회색 범위
        self.light_gray_range = (165, 255)  # 밝은 회색 범위
        self.medium_gray_range = (50, 150)  # 중간 회색 범위
        self.light_medium_gray_range = (110, 210)  # 밝은 중간 회색 범위
        self.faded_color_probability = 0.4  # 흐린 색상 사용 확률

        # 불완전한 선 그리기 설정
        self.enable_imperfect_lines = False  # 불완전한 선 사용 여부
        self.imperfect_line_probability = {
            'top': 1,    # 상단 선 그리기 확률
            'bottom': 1, # 하단 선 그리기 확률
            'left': 1,  # 좌측 선 그리기 확률
            'right': 1  # 우측 선 그리기 확률
        }

        # 회색 셀 설정
        self.enable_gray_cells = False  # 회색 셀 사용 여부
        self.gray_cell_probability = 0.0  # 회색 셀 생성 확률
        self.gray_color_range = (50, 220)  # 회색 색상 범위
        self.no_border_gray_cell_probability = 0.0  # 회색 셀의 테두리 없음 확률

        # 스타일 설정
        self.styles = ['thin', 'medium', 'thick', 'double']  # 선 스타일 옵션
        self.fonts = ['fonts/NanumGothic.ttf', 'fonts/SANGJU Dajungdagam.ttf', 'fonts/SOYO Maple Regular.ttf']  # 사용 가능한 폰트

        # 불완전성 및 효과 설정
        self.enable_table_imperfections = True  # 테이블 불완전성 사용 여부
        self.imperfect_ratio = 0.3  # 불완전성 비율
        self.enable_cell_inside_imperfections = True  # 셀 내부 불완전성 사용 여부
        self.enable_random_lines = True  # 랜덤 선 사용 여부
        self.random_line_probability = 0.3  # 랜덤 선 생성 확률
        self.cell_imperfection_probability = 0.3  # 셀 불완전성 확률

        # 특수 효과 설정
        self.enable_noise = True  # 노이즈 효과 사용 여부
        self.noise_intensity_range = (0.01, 0.05)  # 노이즈 강도 범위
        self.enable_blur = True  # 블러 효과 사용 여부
        self.blur_radius_range = (0.3, 1.0)  # 블러 반경 범위
        self.blur_probability = 0.3  # 블러 적용 확률
        self.enable_brightness_variation = True  # 밝기 변화 사용 여부
        self.enable_contrast_variation = True  # 대비 변화 사용 여부
        self.brightness_factor_range = (0.7, 1.3)  # 밝기 변화 범위
        self.contrast_factor_range = (0.7, 1.3)  # 대비 변화 범위
        self.enable_perspective_transform = False  # 원근 변환 사용 여부
        self.perspective_transform_range = (0.05, 0.17)  # 원근 변환 범위
        self.perspective_intensity = 0.1  # 원근 변환 강도
        self.perspective_direction = None  # 원근 변환 방향
        self.enable_shadow = True  # 그림자 효과 사용 여부
        self.shadow_opacity_range = (30, 150)  # 그림자 불투명도 범위
        self.shadow_blur_radius = 5  # 그림자 블러 반경
        self.shadow_size_ratio = 0.1  # 그림자 크기 비율
        self.shadow_gradient_strength = 1.0  # 그림자 그라데이션 강도
        self.table_crop_probability = 0.0 # 표 자를 확률, 잘라낼, 자르기, 잘라

        # 기타 설정
        self.empty_cell_ratio = 0.1  # 빈 셀 비율
        self.enable_background_shapes = False  # 배경 도형 사용 여부
        self.background_shape_count = 20  # 배경에 추가할 도형 수
        self.line_break_probability = 0.1  # 줄 바꿈 확률
        self.line_thickness_variation_probability = 0.1  # 선 두께 변화 확률
        self.corner_imperfection_probability = 0.1  # 모서리 불완전

        # 텍스처 및 불완전성 효과
        self.texture_effect_probability = 0.1  # 텍스처 효과 적용 확률
        self.line_blur_probability = 0.2  # 선 흐림 효과 적용 확률
        self.irregular_thickness_probability = 0.0  # 불규칙한 선 두께 적용 확률
        self.line_curve_probability = 0.3  # 곡선 효과 적용 확률
        self.color_variation_probability = 0.2  # 색상 변화 적용 확률
        self.end_imperfection_probability = 0.3  # 선 끝부분 불완전성 적용 확률
        self.transparency_variation_probability = 0.2  # 투명도 변화 적용 확률
        self.cell_shift_down_probability = 0.7 # 셀이 선을 뒤덮을 확률
        # 기타 설정
        self.empty_cell_ratio = 0.1  # 빈 셀의 비율
        self.enable_background_shapes = False  # 배경 도형 사용 여부
        self.background_shape_count = 20  # 배경에 추가할 도형의 수

        # 선 스타일 및 폰트 설정
        self.styles = ['thin', 'medium', 'thick', 'double']  # 사용 가능한 선 스타일
        self.fonts = ['fonts/NanumGothic.ttf', 'fonts/SANGJU Dajungdagam.ttf', 'fonts/SOYO Maple Regular.ttf']  # 사용 가능한 폰트

        # 셀 내용 유형 및 텍스트 위치 설정
        self.cell_content_types = ['text', 'shapes', 'mixed']  # 셀 내용 유형
        self.text_positions = ['center', 'top_left', 'top_right', 'bottom_left', 'bottom_right']  # 텍스트 위치 옵션

        # 텍스트 생성 관련 설정
        self.enable_text_generation = True  # 텍스트 생성 사용 여부
        self.max_text_length = 50  # 최대 텍스트 길이
        self.class_info_probability = 0.4  # 클래스 정보 포함 확률
        self.common_word_probability = 0.5  # 일반 단어 사용 확률

        # 도형 관련 설정
        self.enable_shapes = True  # 도형 사용 여부
        self.shape_types = ['rectangle', 'circle', 'triangle', 'line', 'arc', 'polygon']  # 사용 가능한 도형 유형
        self.min_shape_size, self.max_shape_size = 5, 50  # 도형 크기 범위
        self.min_shapes, self.max_shapes = 1, 5  # 도형 개수 범위
        self.shape_line_width = 2  # 도형 선 두께



        # 불완전한 선 그리기 설정
        self.enable_imperfect_lines = False  # 불완전한 선 사용 여부
        self.imperfect_line_probability = {
            'top': 0.9,    # 90% 확률로 상단 선 그리기
            'bottom': 0.9, # 90% 확률로 하단 선 그리기
            'left': 0.95,  # 95% 확률로 좌측 선 그리기
            'right': 0.95  # 95% 확률로 우측 선 그리기
        }

        # 회색 셀 설정
        self.enable_gray_cells = False  # 회색 셀 사용 여부
        self.gray_cell_probability = 0.0  # 회색 셀 생성 확률
        self.gray_color_range = (50, 220)  # 회색 색상 범위
        self.no_border_gray_cell_probability = 0.0  # 회색 셀의 테두리 없음 확률



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

        # 내용 관련
        self.enable_title = False
        self.enable_text_generation = False
        self.enable_text_wrapping = False
        self.enable_shapes = False
        self.enable_overflow = False

        # 색상 관련
        self.enable_gray_cells = False
        

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
        self.enable_background_shapes = False
        self.enable_imperfect_lines = False

        # 확률 관련 설정을 0으로 설정
        self.cell_no_border_probability = 0.0
        self.rounded_corner_probability = 0.0
        self.outer_border_probability = 0.0
        self.overflow_probability = 0.0

        self.gray_cell_probability = 0.0
        self.random_line_probability = 0.0
        self.cell_imperfection_probability = 0.0
        self.blur_probability = 0.0
        self.empty_cell_ratio = 0.0
        self.merged_cell_probability = 0.0
        self.horizontal_merge_probability = 0.0
        self.vertical_merge_probability = 0.0
        self.table_crop_probability = 0
        self.table_side_line_probability = 0.0
        self.rounded_table_corner_probability = 0.0
        self.horizontal_divider_probability = 0.0
        self.vertical_divider_probability = 0.0
        self.no_border_gray_cell_probability = 0.0
        
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


    def test_disable_all_effects(self):
        self.disable_all_effects()
        
        # 모든 enable_ 설정이 False인지 확인
        for attr_name in dir(self):
            if attr_name.startswith('enable_'):
                assert getattr(self, attr_name) == False, f"{attr_name} should be False"
        
        # 모든 확률 관련 설정이 0인지 확인
        probability_attrs = [
            'cell_no_border_probability', 'rounded_corner_probability', 'outer_border_probability',
            'overflow_probability', 'gray_cell_probability', 
            'random_line_probability', 'cell_imperfection_probability', 'blur_probability',
            'empty_cell_ratio', 'merged_cell_probability',
            'horizontal_merge_probability', 'vertical_merge_probability', 'table_side_line_probability',
            'rounded_table_corner_probability', 'horizontal_divider_probability',
            'vertical_divider_probability', 'no_border_cell_probability',
            'no_border_gray_cell_probability', 'faded_color_probability', 'line_break_probability',
            'line_thickness_variation_probability', 'corner_imperfection_probability',
            'texture_effect_probability', 'line_blur_probability', 'irregular_thickness_probability',
            'line_curve_probability', 'color_variation_probability', 'end_imperfection_probability',
            'transparency_variation_probability', 'imperfect_ratio', 'class_info_probability',
            'common_word_probability'
        ]
        for attr_name in probability_attrs:
            assert getattr(self, attr_name) == 0.0, f"{attr_name} should be 0.0"
        
        # 기타 설정 확인
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


    def apply_simple_config(self):
        self.disable_all_effects()
        self.enable_simple_settings()
        self.randomize_simple_settings()
        
    def randomize_settings(self):
        if self.image_level == 0:
            self.disable_all_effects()
        table_types = ['no_header', 'header_row', 'header_column', 'header_both']  # 가능한 테이블 유형들
        
        
        self.table_type = random.choice(table_types)  # 랜덤으로 테이블 유형 선택

        if self.image_level >= 1.5:
            self.table_type = random.choice(['no_header', 'header_row', 'header_column', 'header_both'])
        else:
            self.table_type = 'no_header'

        # 이미지 크기 랜덤화
        self.min_image_width = random.randint(600, 1000)
        self.max_image_width = random.randint(self.min_image_width, 2800)
        self.min_image_height = random.randint(600, 1000)
        self.max_image_height = random.randint(self.min_image_height, 2900)



        if self.image_level > 5:

            # 불완전한 선 그리기 설정
            self.enable_imperfect_lines = random.choice([True, False])
            self.imperfect_line_probability = {'top': 1, 'bottom': 1, 'left': 1, 'right': 1} # 선 관련 코드


        else:
            self.enable_imperfect_lines = False
            self.imperfect_line_probability = {'top': 0.4, 'bottom': 0.7, 'left': 0.7, 'right': 0.6} # 선 관련 코드

        
        # 셀 크기 설정 랜덤화
        self.min_cell_width = random.randint(20, 90)
        self.max_cell_width = random.randint(self.min_cell_width, 240)
        self.min_cell_height = random.randint(20, 90)
        self.max_cell_height = random.randint(self.min_cell_height, 240)

        # 셀 간격 설정 랜덤화
        self.min_cell_gap = random.randint(1, 2)
        self.max_cell_gap = random.randint(self.min_cell_gap, 3)
        self.enable_cell_gap = random.choices([True, False], weights=[3, 7])[0]

        if self.image_level > 5:
            self.no_side_borders_cells_probability = 0.2# 선 제거
        else:
            self.no_side_borders_cells_probability = 0
        # 테이블 구조 랜덤화
        self.min_table_width = random.randint(200, 400)
        self.min_table_height = random.randint(200, 400)
        self.min_cols = random.randint(3, 6)
        self.max_cols = random.randint(self.min_cols, 15) # 최대 일주일치 + 3
        self.min_rows = random.randint(3, 6)
        self.max_rows = random.randint(self.min_rows, 15) 
        self.set_table_dimensions()

        # 둥근 모서리 설정 랜덤화
        self.enable_rounded_corners = random.choice([True, False])
        if self.enable_rounded_corners:
            self.rounded_corner_probability = random.uniform(0.1, 0.4)
            self.min_corner_radius = random.randint(1, 5)
            self.max_corner_radius = random.randint(self.min_corner_radius, 25)

        # 선 제거 코드, 셀 사이드 제거하는 거랑 겹치면 어색할지도
        # 테이블 측면 선 설정, 이 값보다 높게 나오면 선 안 그림
        self.table_side_line_probability = random.uniform(0.1, 0.3) # 선 설정

        # 헤더 설정 랜덤화
        self.header_row_height_factor = random.uniform(0.2, 1.3)
        self.header_col_width_factor = random.uniform(0.2, 1.3)

        # 선 스타일 설정 랜덤화
        self.irregular_thickness_probability = random.uniform(0.1, 0.2)
        self.line_curve_probability = random.uniform(0.2, 0.6)
        self.color_variation_probability = random.uniform(0.2, 0.7)
        self.end_imperfection_probability = random.uniform(0.5, 0.7)
        self.dotted_line_probability = random.uniform(0.5,0.7)
        self.transparency_variation_probability = random.uniform(0.1, 0.3)
        self.min_line_thickness = random.randint(1, 2)
        self.max_line_thickness = random.randint(self.min_line_thickness + 2, 4)  # 최대 선 두께

        # 구분선 설정 랜덤화
        self.horizontal_divider_probability = random.uniform(0.2, 0.3)
        self.vertical_divider_probability = random.uniform(0.2, 0.3)
        self.enable_divider_lines = random.choice([True, False])
        self.divider_line_thickness_range = (2, 6)  # 구분선 두께 범위
        
        # 여백 및 테두리 설정 랜덤화
        self.min_margin = random.randint(0, 200)
        self.max_margin = random.randint(self.min_margin, 1200)
        self.enable_outer_border = random.choice([True, False])
        self.outer_border_probability = random.uniform(0.6, 0.8)
        self.min_outer_line_thickness = random.randint(1, 2)
        self.max_outer_line_thickness = random.randint(self.min_outer_line_thickness + 1, 5)

         # 색상 설정 랜덤화
        self.color_brightness_threshold = random.randint(100, 150)
        self.dark_gray_range = (random.randint(0, 32), random.randint(33, 96))
        self.light_gray_range = (random.randint(130, 224), 255)
        self.medium_gray_range = (random.randint(32, 96), random.randint(66, 160))
        self.light_medium_gray_range = (random.randint(96, 160), random.randint(121, 224))
        self.faded_color_probability = random.uniform(0.1, 0.4)
        
        # 컬러 셀 및 회색 셀 설정 랜덤화
        
        self.enable_gray_cells = random.choices([True, False], weights=[7, 3])[0]
        if self.enable_gray_cells:
            self.gray_cell_probability = random.uniform(0.5, 0.7)
            min_gray = random.randint(50, 100)
            max_gray = random.randint(min_gray + 30, 250)
            self.gray_color_range = (min_gray, max_gray)
        
        # 제목 설정 랜덤화
        self.enable_title = random.choice([True, False])
        self.min_title_size = random.randint(15, 25)
        self.max_title_size = random.randint(self.min_title_size, 50)

        # 셀 내용 설정 랜덤화
        self.enable_text_generation = random.choice([True, False])
        self.enable_text_wrapping = random.choice([True, False])
        self.max_text_length = random.randint(60, 200)
        self.class_info_probability = random.uniform(0.4, 0.7)
        self.common_word_probability = random.uniform(0.4, 0.7)
        if self.image_level > 2.5:
            # 오버플로우 설정 랜덤화
            self.enable_overflow = random.choices([True, False], weights=[6, 4])[0]
            self.overflow_probability = random.uniform(0.3, 0.7)
            self.min_overflow_height = random.randint(1, min(10, self.max_cell_height - 10))
            if self.max_cell_height > self.min_overflow_height:
                self.max_overflow_height = random.randint(self.min_overflow_height + 1, self.max_cell_height - 1)
            else:
                self.max_overflow_height = self.max_cell_height - 15
        elif self.image_level > 2:
            self.enable_overflow = random.choices([True, False], weights=[3, 7])[0]
            self.overflow_probability = random.uniform(0.3, 0.4)
            self.min_overflow_height = random.randint(25, 30)
        else:
            self.enable_overflow = False
        # 도형 설정 랜덤화
        self.enable_shapes = random.choice([True, False])
        self.min_shapes = random.randint(10, 15)
        self.max_shapes = random.randint(self.min_shapes, 45)
        self.shape_line_width = random.randint(1, 10)

        if self.image_level > 2.5:

            # 셀 병합 설정 랜덤화
            self.enable_cell_merging = random.choices([True, False], weights=[8, 2])[0]
            self.enable_horizontal_merge = random.choices([True, False], weights=[5,5])[0]
            self.enable_vertical_merge = random.choices([True, False], weights=[6, 4])[0]
            self.merged_cell_ratio = random.uniform(0.3, 0.7)
            self.merged_cell_probability = random.uniform(0.3, 0.6)
            self.max_horizontal_merge = random.randint(2, 7)
            self.max_vertical_merge = random.randint(2, 7)
            self.horizontal_merge_probability = random.uniform(0.3, 0.6)
            self.vertical_merge_probability = random.uniform(0.3, 0.6)
        elif self.image_level > 2:
            # 셀 병합 설정 랜덤화
            self.enable_cell_merging = random.choices([True, False], weights=[3, 7])[0]
            self.enable_horizontal_merge = random.choices([True, False], weights=[3,7])[0]
            self.enable_vertical_merge = random.choices([True, False], weights=[3, 7])[0]
            self.merged_cell_ratio = random.uniform(0.3, 0.4)
            self.merged_cell_probability = random.uniform(0.2, 0.4)
            self.max_horizontal_merge = random.randint(2, 3)
            self.max_vertical_merge = random.randint(2, 4)
            self.horizontal_merge_probability = random.uniform(0.3, 0.4)
            self.vertical_merge_probability = random.uniform(0.3, 0.7)
        else:
            self.enable_cell_merging = False
        
       

        # 불완전성 및 효과 설정 랜덤화  
        self.enable_table_imperfections = random.choices([True, False], weights=[6, 4])[0]
        self.imperfect_ratio = random.uniform(0.1, 0.6)
        self.enable_random_lines = random.choices([True, False], weights=[6,4])[0]
        self.enable_cell_inside_imperfections = random.choice([True, False])
        self.random_line_probability = 0.1
        if self.image_level > 5:
            self.cell_imperfection_probability = random.uniform(0.3, 0.7)
            self.table_crop_probability = random.uniform(0.1, 0.3)
            self.line_break_probability = random.uniform(0.3, 0.6)  # 줄 바꿈 확률
            self.corner_imperfection_probability = random.uniform(0.1, 0.3)  # 모서리 불완전
            self.line_blur_probability = random.uniform(0.3, 0.6)
            self.cell_shift_down_probability = random.uniform(0.1, 0.3)
            self.no_border_gray_cell_probability = random.uniform(0.1, 0.2)
            self.cell_no_border_probability = random.uniform(0.1, 0.2) # 선 제거, 선 삭제, 선 안 그리기
        else:
            self.cell_imperfection_probability = 0
            self.table_crop_probability = 0
            
            
        # 특수 효과 설정 랜덤화
        self.randomize_noise_effect()
        self.randomize_blur_effect()
        self.randomize_brightness_contrast_effect()
        self.randomize_shadow_effect()


        # 기타 설정 랜덤화
        self.empty_cell_ratio = random.uniform(0.1, 0.2)
        self.enable_background_shapes = random.choice([True, False])

    def get_random_line_thickness(self):
        thickness_type = random.choices(list(self.line_thickness_distribution.keys()),
                                        weights=list(self.line_thickness_distribution.values()))[0]
        if thickness_type == 'thin':
            return max(1, random.randint(self.min_line_thickness, max(2, self.min_line_thickness + 1)))
        elif thickness_type == 'normal':
            return max(1, random.randint(min(3, self.max_line_thickness - 1), max(3, self.max_line_thickness - 1)))
        else:  # thick
            return max(1, random.randint(min(4, self.max_line_thickness), self.max_line_thickness))
    def randomize_noise_effect(self):
        self.enable_noise = random.choices([True, False], weights=[6, 4])[0]
        if self.enable_noise:
            self.noise_intensity_range = (random.uniform(0.01, 0.05), random.uniform(0.05, 0.1))

    def randomize_blur_effect(self):
        self.enable_blur = random.choices([True, False], weights=[6, 4])[0]
        if self.enable_blur:
            self.blur_radius_range = (random.uniform(0.3, 0.6), random.uniform(0.7, 1.3))
            self.blur_probability = random.uniform(0.1, 0.4)

    def randomize_brightness_contrast_effect(self):
        self.enable_brightness_variation = random.choices([True, False], weights=[6, 4])[0]
        self.enable_contrast_variation = random.choices([True, False], weights=[6, 4])[0]
        if self.enable_brightness_variation:
            self.brightness_factor_range = (random.uniform(0.4, 0.7), random.uniform(0.7, 1.0))
        if self.enable_contrast_variation:
            self.contrast_factor_range = (random.uniform(0.4, 0.7), random.uniform(0.7, 1.0))

    def randomize_shadow_effect(self):
        self.enable_shadow = random.choices([True, False], weights=[6, 4])[0]
        if self.enable_shadow:
            self.shadow_opacity_range = (random.randint(10, 30), random.randint(40, 120))
            self.shadow_blur_radius = random.uniform(1, 25)
            self.shadow_size_ratio = random.uniform(0.05, 0.25)



    def get_random_gray_color(self, bg_color):
        if not self.enable_gray_cells:
            return None

        bg_brightness = sum(bg_color) / 3

        # 배경 밝기에 따라 적절한 회색 선택
        if bg_brightness > 200:
            base_gray = self.predefined_grays[0]  # 매우 어두운 회색
        elif bg_brightness > 150:
            base_gray = self.predefined_grays[1]  # 어두운 회색
        elif bg_brightness > 100:
            base_gray = self.predefined_grays[2]  # 중간 회색
        elif bg_brightness > 50:
            base_gray = self.predefined_grays[3]  # 밝은 회색
        else:
            base_gray = self.predefined_grays[4]  # 매우 밝은 회색

        # 선택된 회색에 변동 추가
        gray_value = max(0, min(255, base_gray + random.randint(-self.gray_variation, self.gray_variation)))

        return (gray_value, gray_value, gray_value)

config = TableGenerationConfig()


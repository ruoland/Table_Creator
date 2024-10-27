import random
from typing import Dict
from typing import Tuple
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

    min_line_thickness: int = 1
    max_line_thickness: int = 5
    line_thickness_distribution: Dict[str, float] = {
        'thin': 0.4,
        'normal': 0.4,
        'thick': 0.2
    }

    def setting_table(self):
        # 이미지 크기 및 해상도 설정
        self.min_image_width, self.max_image_width = 1200, 4000  # 이미지 너비 범위
        self.min_image_height, self.max_image_height = 1200, 4000  # 이미지 높이 범위
        self.predefined_grays = [
            50,   # 매우 어두운 회색
            80,   # 어두운 회색
            70,  # 중간 회색
            80,  # 밝은 회색
            100   # 매우 밝은 회색
        ]
        self.gray_variation = 20  # 회색 값의 변동 범위
        # 테이블 구조 설정
        self.min_cols, self.max_cols = 4, 10  # 열 수 범위
        self.min_rows, self.max_rows = 4, 10  # 행 수 범위
        self.min_table_width, self.min_table_height = 200, 100  # 최소 테이블 크기
        # 여백 설정
        self.min_margin, self.max_margin = 10, 100  # 여백 범위
 
    def setting_cells(self):
        self.min_cell_width, self.max_cell_width = 5, 200  # 셀 너비 범위
        self.min_cell_height, self.max_cell_height = 5, 300  # 셀 높이 범위
        self.enable_cell_gap = True  # 셀 간격 사용 여부
        self.min_cell_gap, self.max_cell_gap = 0, 3  # 셀 간격 범위
        self.cell_no_border_probability = 0  # 셀에 테두리를 그리지 않을 확률
    # 회색 셀 설정
        self.enable_gray_cells = False  # 회색 셀 사용 여부
        self.gray_cell_probability = 0.0  # 회색 셀 생성 확률
        self.gray_color_range = (50, 220)  # 회색 색상 범위
        self.no_border_gray_cell_probability = 0.0  # 회색 셀의 테두리 없음 확률
        self.empty_cell_ratio = 0.1  # 빈 셀 비율

                # 텍스트 생성 관련 설정
        self.enable_text_generation = True  # 텍스트 생성 사용 여부
        self.max_text_length = 50  # 최대 텍스트 길이
        self.class_info_probability = 0.4  # 클래스 정보 포함 확률
        self.common_word_probability = 0.5  # 일반 단어 사용 확률
        
                # 셀 내용 유형 및 텍스트 위치 설정
        self.cell_content_types = ['text', 'shapes', 'mixed']  # 셀 내용 유형
        self.text_positions = ['center', 'top_left', 'top_right', 'bottom_left', 'bottom_right']  # 텍스트 위치 옵션


    def setting_cells_content(self):
         # 셀 내용 설정
        self.cell_content_types = ['text', 'shapes', 'mixed']  # 셀 내용 유형
        self.text_positions = ['center', 'top_left', 'top_right', 'bottom_left', 'bottom_right']  # 텍스트 위치 옵션
        self.enable_text_generation = True  # 텍스트 생성 사용 여부
        self.max_text_length = 50  # 최대 텍스트 길이
        self.class_info_probability = 0  # 클래스 정보 포함 확률
        self.common_word_probability = 0.5  # 일반 단어 사용 확률
        self.empty_cell = False
        self.empty_cell_probability = 0.3
    def setting_cells_overflow(self):
         # 오버플로우 설정
        self.enable_overflow = True  # 오버플로우 사용 여부
        self.overflow_probability = 0.7  # 오버플로우 발생 확률
        self.min_overflow_height = 10  # 최소 오버플로우 높이
        self.max_overflow_height = 10  # 오버플로우 영향을 확인할 최대 행 수
        self.overflow_only_row_probability = 0.1
        self.default_row_height = 30
        self.default_col_width = 100
        self.row_overflow_probability = 0.4
        self.overflow_calc = random.uniform(0.3, 0.8)

    def setting_lines(self):
        self.enable_divider_lines = True  # 구분선 사용 여부
        self.horizontal_divider_probability = 0.2  # 수평 구분선 확률
        self.vertical_divider_probability = 0.2  # 수직 구분선 확률
        self.divider_line_thickness_range = (2, 5)  # 구분선 두께 범위
        self.enable_outer_border = True  # 외곽선 사용 여부
        self.outer_border_probability = 0.8  # 외곽선 그릴 확률
        self.min_line_thickness = max(1, random.randint(1, 3))  # 최소 선 두께
        self.max_line_thickness = max(1, random.randint(self.min_line_thickness + 2, 5))  # 최대 선 두께
        self.min_outer_line_thickness, self.max_outer_line_thickness = 1, 4  # 외곽선 두께 범위
        # 스타일 설정
        self.styles = ['thin', 'medium', 'thick', 'double']  # 선 스타일 옵션
    def setting_rounded(self):
        # 모서리 및 형태 설정
        self.enable_rounded_corners = True  # 둥근 모서리 사용 여부
        self.rounded_corner_probability = 0.7  # 둥근 모서리 확률
        self.min_corner_radius = 2  # 최소 모서리 반경
        self.max_corner_radius = 7  # 최대 모서리 반경
        self.enable_rounded_table_corners = True  # 테이블 모서리 둥글게 처리 여부
        self.rounded_table_corner_probability = 0.7  # 테이블 모서리 둥글게 처리 확률
        self.min_table_corner_radius = 5  # 최소 테이블 모서리 반경
        self.max_table_corner_radius = 24  # 최대 테이블 모서리 반경
        
    def setting_headers(self):# 헤더 설정
        # 제목 설정
        self.enable_title = True  # 제목 사용 여부
        self.min_title_size, self.max_title_size = 20, 40  # 제목 크기 범위
        self.header_row_height_factor = 0.9  # 헤더 행 높이 배수
        self.header_col_width_factor = 0.9  # 헤더 열 너비 배수
        self.no_side_borders_cells_probability = 0.2 # 양옆 선 없는 셀의 확률
        
    def setting_imperfect(self):
        
                # 기타 설정
        self.line_break_probability = 0.1  # 선의 끊김 효과
        self.line_thickness_variation_probability = 0.1  # 선 두께 변화 확률
        
        # 텍스처 및 불완전성 효과
        self.texture_effect_probability = 0.1  # 텍스처 효과 적용 확률
        self.line_blur_probability = 0.2  # 선 흐림 효과 적용 확률
        self.irregular_thickness_probability = 0.0  # 불규칙한 선 두께 적용 확률
        self.line_curve_probability = 0.3  # 곡선 효과 적용 확률
        self.color_variation_probability = 0.2  # 색상 변화 적용 확률
        self.end_imperfection_probability = 0.3  # 선 끝부분 불완전성 적용 확률
        self.transparency_variation_probability = 0.2  # 투명도 변화 적용 확률
        self.cell_shift_down_probability = 0.7 # 셀이 선을 뒤덮을 확률
        
        
        # 불완전한 선 그리기 설정
        self.enable_imperfect_lines = False  # 불완전한 선 사용 여부
        self.imperfect_line_probability = {
            'top': 1,    # 상단 선 그리기 확률
            'bottom': 1, # 하단 선 그리기 확률
            'left': 1,  # 좌측 선 그리기 확률
            'right': 1  # 우측 선 그리기 확률
        }
    def setting_cell_merging(self):
        # 셀 병합 설정
        self.enable_cell_merging = True  # 셀 병합 사용 여부
        self.merged_cell_probability = 0.5  # 셀 병합 확률
        self.max_horizontal_merge = 4  # 최대 수평 병합 수
        self.max_vertical_merge = 4  # 최대 수직 병합 수
        self.horizontal_merge_probability = 0.3  # 수평 병합 확률
        self.vertical_merge_probability = 0.3  # 수직 병합 확률
        self.enable_horizontal_merge = random.choices([True, False], weights=[7,3])[0]
        self.enable_vertical_merge = random.choices([True, False], weights=[7, 3])[0]
    def setting_color(self):
                
        # 색상 설정
        self.background_colors = {
            'light': {'white': (255, 255, 255)},
            'dark': {'black': (0, 0, 0)}
        }  # 배경색 옵션

    def setting_table_imperfection(self):
            # 불완전성 및 효과 설정
        self.enable_table_imperfections = True  # 테이블 불완전성 사용 여부
        self.imperfect_ratio = 0.3  # 불완전성 비율
        self.enable_cell_inside_imperfections = True  # 셀 내부 불완전성 사용 여부
        self.enable_random_lines = True  # 랜덤 선 사용 여부
        self.random_line_probability = 0.3  # 랜덤 선 생성 확률
        self.cell_imperfection_probability = 0.3  # 셀 불완전성 확률
        
    def setting_table_special(self):
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
        self.background_shape_count = 20  # 배경에 추가할 도형 수
        
        # 도형 설정
        self.enable_shapes = True  # 도형 사용 여부
        self.shape_types = ['rectangle', 'circle', 'triangle', 'line', 'arc', 'polygon']  # 사용 가능한 도형 유형
        self.min_shape_size, self.max_shape_size = 5, 50  # 도형 크기 범위
        self.min_shapes, self.max_shapes = 1, 5  # 도형 개수 범위
        self.shape_line_width = 2  # 도형 선 두께
    def __init__(self):
        
        # 테이블 유형 및 기본 설정
        table_types = ['no_header', 'header_row', 'header_column', 'header_both']  # 가능한 테이블 유형들
        self.table_type = random.choice(table_types)  # 랜덤으로 테이블 유형 선택
        self.config_mode = 'Random'  # 설정 모드 (None 또는 Random)
        self.total_images = 5000  # 생성할 총 이미지 수
        self.image_level = 3.1 # 1은 셀, 표, 행, 열만 탐지하고, 2는 헤더 행 열, 2.6는 적은 수의 병합 셀, 오버플로, 3.1부터는 병합된 셀과 오버플로우 약간씩, 4.1부터는 병합된 셀과 오버플로우 셀 많이, 5.1부터는 불완전한 표, 선 없는 셀, 색깔 셀에서 선 안 그리기, 등.
        self.total_rows = None  # 실제 행 수 (자동 설정)
        self.total_cols = None  # 실제 열 수 (자동 설정)
        self.fonts = ['fonts/NanumGothic.ttf', 'fonts/SANGJU Dajungdagam.ttf', 'fonts/SOYO Maple Regular.ttf']  # 사용 가능한 폰트
        self.simple_ratio = 0.2
        self.middle_ratio = 0.7
        self.complex_ratio = 0.1
        self.setting_table()
        self.setting_cells()
        self.setting_lines()
        self.setting_rounded()
        self.setting_headers()
        self.setting_cells_content()
        self.setting_cells_overflow()
        self.setting_imperfect()
        self.setting_cell_merging()
        self.setting_color()
        self.setting_table_imperfection()
        self.setting_table_special()
        self.corner_imperfection_probability = 0.1  # 모서리 불완전
        # 선 스타일 및 폰트 설정
        self.styles = ['thin', 'medium', 'thick', 'double']  # 사용 가능한 선 스타일
        self.fonts = ['fonts/NanumGothic.ttf', 'fonts/SANGJU Dajungdagam.ttf', 'fonts/SOYO Maple Regular.ttf']  # 사용 가능한 폰트

    def set_table_dimensions(self):
        """실제 테이블 크기를 설정합니다."""
        self.total_rows = random.randint(self.min_rows, self.max_rows)
        self.total_cols = random.randint(self.min_cols, self.max_cols)


    def randomize_settings(self):
        if self.image_level == 0:
            self.disable_all_effects()
        table_types = ['no_header', 'header_row', 'header_column', 'header_both']  # 가능한 테이블 유형들
        
        
        self.table_type = random.choice(table_types)  # 랜덤으로 테이블 유형 선택

        if self.image_level >= 1:
            self.table_type = random.choice(['no_header', 'header_row', 'header_column', 'header_both'])
        else:
            self.table_type = 'no_header'

        # 이미지 크기 랜덤화
        self.min_image_width = random.randint(600, 1000)
        self.max_image_width = random.randint(self.min_image_width, 2800)
        self.min_image_height = random.randint(600, 1000)
        self.max_image_height = random.randint(self.min_image_height, 2600)



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
        self.min_cell_gap = random.randint(2, 3)
        self.max_cell_gap = random.randint(self.min_cell_gap, 4)
        self.enable_cell_gap = random.choices([True, False], weights=[4, 6])[0]

        if self.image_level > 5:
            self.no_side_borders_cells_probability = 0.2# 선 제거
        else:
            self.no_side_borders_cells_probability = 0
        # 테이블 구조 랜덤화
        self.min_table_width = random.randint(200, 400)
        self.min_table_height = random.randint(200, 400)
        self.min_cols = random.randint(3, 6)
        self.max_cols = random.randint(self.min_cols, 12) # 최대 일주일치 + 3
        self.min_rows = random.randint(3, 6)
        self.max_rows = random.randint(self.min_rows, 12) 
        self.set_table_dimensions()

        # 둥근 모서리 설정
        self.enable_rounded_corners = random.choices([True, False], weights=[5, 5])[0]
        if self.enable_rounded_corners:
            self.rounded_corner_probability = random.uniform(0.4, 0.7)
            self.min_corner_radius = random.randint(15, 25)
            self.max_corner_radius = random.randint(self.min_corner_radius, 45)

        # 선 제거 코드, 셀 사이드 제거하는 거랑 겹치면 어색할지도
        # 테이블 측면 선 설정, 이 값보다 높게 나오면 선 안 그림
        self.table_side_line_probability = random.uniform(0.1, 0.3) # 선 설정

        # 헤더 설정 랜덤화
        self.header_row_height_factor = random.uniform(0.2, 0.7)
        self.header_col_width_factor = random.uniform(0.2, 0.7)

        # 선 스타일 설정 랜덤화
        self.irregular_thickness_probability = random.uniform(0.1, 0.5)
        self.line_curve_probability = random.uniform(0.3, 0.7)
        self.color_variation_probability = random.uniform(0.4, 0.7)
        self.end_imperfection_probability = random.uniform(0.5, 0.7)
        self.dotted_line_probability = random.uniform(0.5,0.7)
        self.transparency_variation_probability = random.uniform(0.2, 0.5)
        self.min_line_thickness = random.randint(1, 2)
        self.max_line_thickness = random.randint(self.min_line_thickness + 2, 6)  # 최대 선 두께

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

        
        self.enable_gray_cells = random.choices([True, False], weights=[6, 4])[0]
        if self.enable_gray_cells:
            self.gray_cell_probability = random.uniform(0.3, 0.5)
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
            self.enable_overflow = random.choices([True, False], weights=[6, 3])[0]
            self.overflow_probability = random.uniform(0.1, 0.2)
            self.overflow_calc = random.uniform(0.3, 0.5)
            self.min_overflow_height = random.randint(5, min(10, self.max_cell_height - 10))

            if self.max_cell_height > self.min_overflow_height:
                self.max_overflow_height = random.randint(self.min_overflow_height + 1, self.max_cell_height - 5)
            else:
                self.max_overflow_height = self.max_cell_height / 2

        elif self.image_level > 2:
            self.enable_overflow = random.choices([True, False], weights=[3, 7])[0]
            self.overflow_probability = random.uniform(0.3, 0.4)
            self.min_overflow_height = random.randint(25, 30)
        else:
            self.enable_overflow = False
        # 도형 설정 랜덤화
        self.enable_shapes = random.choice([True, False])
        self.min_shapes = random.randint(10, 15)
        self.max_shapes = random.randint(self.min_shapes, 75)
        self.shape_line_width = random.randint(1, 25)

        if self.image_level > 2.5:
            # 셀 병합 설정 랜덤화
            self.enable_cell_merging = random.choices([True, False], weights=[8, 2])[0]
            self.enable_horizontal_merge = random.choices([True, False], weights=[7,3])[0]
            self.enable_vertical_merge = random.choices([True, False], weights=[7, 3])[0]
            self.merged_cell_probability = random.uniform(0.1, 0.3)
            self.max_horizontal_merge = random.randint(2, 5)
            self.max_vertical_merge = random.randint(2, 5)
            self.horizontal_merge_probability = random.uniform(0.2, 0.3)
            self.vertical_merge_probability = random.uniform(0.2, 0.3)
        elif self.image_level > 2:
            # 셀 병합 설정 랜덤화
            self.enable_cell_merging = random.choices([True, False], weights=[3, 7])[0]
            self.merged_cell_probability = random.uniform(0.2, 0.4)
            self.max_horizontal_merge = random.randint(2, 3)
            self.max_vertical_merge = random.randint(2, 4)
            self.horizontal_merge_probability = random.uniform(0.3, 0.4)
            self.vertical_merge_probability = random.uniform(0.3, 0.3)
        else:
            self.enable_cell_merging = False
        
       

        # 불완전성 및 효과 설정 랜덤화  
        self.enable_table_imperfections = random.choices([True, False], weights=[6, 4])[0]
        self.imperfect_ratio = random.uniform(0.2, 0.6)
        self.enable_random_lines = random.choices([True, False], weights=[6,4])[0]
        self.enable_cell_inside_imperfections = random.choice([True, False])
        self.random_line_probability = 0.1
        if self.image_level > 5:
            self.cell_imperfection_probability = random.uniform(0.5, 0.8)
            self.table_crop_probability = random.uniform(0.3, 0.5)
            self.line_break_probability = random.uniform(0.3, 0.6)  # 선 끊김
            
            self.corner_imperfection_probability = random.uniform(0.3, 0.5)  # 모서리 불완전
            self.line_blur_probability = random.uniform(0.3, 0.6)
            self.cell_shift_down_probability = random.uniform(0.3, 0.5)
            self.no_border_gray_cell_probability = random.uniform(0.1, 0.3)
            self.cell_no_border_probability = random.uniform(0.1, 0.3) # 선 제거, 선 삭제, 선 안 그리기
        else:
            self.cell_imperfection_probability = 0
            self.table_crop_probability = 0
            
            
        # 특수 효과 설정 랜덤화
        self.randomize_noise_effect()
        self.randomize_blur_effect()
        self.randomize_brightness_contrast_effect()
        self.randomize_shadow_effect()

        self.overflow_or_merged()
        # 기타 설정 랜덤화
        self.empty_cell_ratio = random.uniform(0.3, 0.5)
    def overflow_or_merged(self):
        self.enable_overflow = random.choice([True,False])
        if not self.enable_overflow:
            self.enable_cell_merging = True
            self.enable_overflow = False

        else:
            self.enable_overflow = True
            self.enable_cell_merging = False

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
            self.blur_radius_range = (random.uniform(0.3, 0.5), random.uniform(0.7, 1.3))
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
            return bg_color

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


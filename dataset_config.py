
import random
# 로깅 설정
import logging

# 루트 로거의 레벨을 설정
logging.getLogger().setLevel(logging.INFO)

# 핸들러를 생성하고 레벨을 설정
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# 포맷터 설정
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# 루트 로거에 핸들러 추가
logging.getLogger().addHandler(handler)
class TableGenerationConfig:
    def __init__(self):
        # 기본 설정들은 그대로 유지
        self.enable_imperfect_lines = False
        self.enable_text_generation = True
        self.enable_shapes = False
        self.enable_cell_merging = True
        self.enable_title = True
        self.enable_background_shapes = False
        self.enable_random_size = True
        self.min_table_width = 400
        self.max_table_width = 1300
        self.min_table_height = 300
        self.max_table_height = 1400
        self.min_cols = 2
        self.max_cols = 10
        self.min_rows = 2
        self.max_rows = 10
        self.min_cell_width = 10
        self.max_cell_width = 400
        self.min_cell_height = 20
        self.max_cell_height = 300
        self.enable_cell_gap = True
        self.min_cell_gap = 0
        self.max_cell_gap = 2
        self.enable_outer_border = True
        self.enable_cell_border = True
        self.table_pattern = 'random'
        self.table_type = 'standard'
        self.enable_overflow_cells = False
        self.overflow_probability = 0.1
        self.max_overflow = 0.3
        # 노이즈 설정 (활성화)
        self.enable_noise = True
        self.noise_intensity_range = (0.01, 0.05)
        self.shadow_gradient_strength = 1.0  # 그라데이션 강도 조절 (0.0 ~ 1.0)
        # 블러 설정 (활성화)
        self.enable_blur = True
        self.blur_radius_range = (0.3, 1.0)
        
        # 밝기 변화 설정 (비활성화)
        self.enable_brightness_variation = False
        self.brightness_factor_range = (1.0, 1.0)
        
        # 대비 변화 설정 (비활성화)
        self.enable_contrast_variation = False
        self.contrast_factor_range = (1.0, 1.0)
        
        # 원근 변환 설정 수정
        self.enable_perspective_transform = True
        self.perspective_transform_range = (0.05, 0.577)  # 최대 30도까지
        self.perspective_intensity = 0.3  # 기본값을 중간 정도로 설정
        self.perspective_direction = None  # None for random, or 'left', 'right', 'top', 'bottom', 'top_left', 'top_right', 'bottom_left', 'bottom_right'
        self.enable_shadow = random.choice([True, False])
        if self.enable_shadow:
            self.shadow_opacity_range = (random.randint(30, 70), random.randint(70, 150))
            self.shadow_opacity = random.randint(*self.shadow_opacity_range)
            self.shadow_blur_radius = random.uniform(5, 15)
            self.shadow_size_ratio = random.uniform(0.05, 0.15)

            self.shadow_size_ratio = 0.1  # 이미지 크기 대비 그림자 크기 비율
        # 표 잘림 설정 (비활성화)
        self.enable_table_cropping = False
        self.table_crop_probability = 0.0
        self.max_crop_ratio = 0.0
        self.enable_perspective_transform = True
    def randomize_settings(self):
        # 노이즈 설정
        self.enable_noise = random.choice([True, False])
        if self.enable_noise:
            self.noise_intensity_range = (random.uniform(0.01, 0.05), random.uniform(0.05, 0.1))
            self.noise_intensity = random.uniform(*self.noise_intensity_range)

        # 블러 설정
        self.enable_blur = random.choice([True, False])
        if self.enable_blur:
            self.blur_radius_range = (random.uniform(0.3, 0.7), random.uniform(0.7, 1.5))
            self.blur_radius = random.uniform(*self.blur_radius_range)

        # 밝기 변화 설정
        self.enable_brightness_variation = random.choice([True, False])
        if self.enable_brightness_variation:
            self.brightness_factor_range = (
                random.uniform(0.7, 1.0),
                random.uniform(1.0, 1.3)
            )

        # 대비 변화 설정
        self.enable_contrast_variation = random.choice([True, False])
        if self.enable_contrast_variation:
            self.contrast_factor_range = (
                random.uniform(0.7, 1.0),
                random.uniform(1.0, 1.3)
            )

        # 원근 변환 설정
        self.enable_perspective_transform = random.choice([True, False])
        if self.enable_perspective_transform:
            self.perspective_transform_range = (
                random.uniform(0.05, 0.2),
                random.uniform(0.2, 0.577)
            )
            self.perspective_intensity = random.uniform(*self.perspective_transform_range)
            self.perspective_direction = random.choice([
                None, 'left', 'right', 'top', 'bottom',
                'top_left', 'top_right', 'bottom_left', 'bottom_right'
            ])

        # 그림자 효과 설정
        self.enable_shadow = random.choice([True, False])
        if self.enable_shadow:
            self.shadow_opacity = random.randint(50, 150)
            self.shadow_blur_radius = random.uniform(5, 15)
            self.shadow_size_ratio = random.uniform(0.05, 0.15)

        # 표 잘림 설정
        self.enable_table_cropping = random.choice([True, False])
        if self.enable_table_cropping:
            self.table_crop_probability = random.uniform(0.1, 0.5)
            self.max_crop_ratio = random.uniform(0.1, 0.3)

        # 기타 설정들도 랜덤화
        self.enable_imperfect_lines = random.choice([True, False])
        self.enable_shapes = random.choice([True, False])
        self.enable_cell_merging = random.choice([True, False])
        self.enable_title = random.choice([True, False])
        self.enable_background_shapes = random.choice([True, False])
        self.enable_random_size = random.choice([True, False])
        self.enable_cell_gap = random.choice([True, False])
        self.enable_outer_border = random.choice([True, False])
        self.enable_cell_border = random.choice([True, False])
        self.enable_overflow_cells = random.choice([True, False])

        if self.enable_random_size:
            self.min_table_width = random.randint(300, 600)
            self.max_table_width = random.randint(800, 1300)
            self.min_table_height = random.randint(200, 400)
            self.max_table_height = random.randint(600, 1400)

        self.min_cols = random.randint(2, 5)
        self.max_cols = random.randint(6, 10)
        self.min_rows = random.randint(2, 5)
        self.max_rows = random.randint(6, 10)

        if self.enable_cell_gap:
            self.min_cell_gap = random.randint(0, 1)
            self.max_cell_gap = random.randint(1, 3)

        if self.enable_overflow_cells:
            self.overflow_probability = random.uniform(0.05, 0.2)
            self.max_overflow = random.uniform(0.1, 0.4)

        self.table_pattern = random.choice(['random', 'checkerboard', 'striped'])
        self.table_type = random.choice(['standard', 'complex'])

    def set_perspective_intensity(self, intensity):
        """원근 변환 강도를 설정합니다. (0.0 ~ 1.0)"""
        min_intensity, max_intensity = self.perspective_transform_range
        self.perspective_intensity = max(min_intensity, min(max_intensity, intensity))

    def set_perspective_transform_range(self, min_value, max_value):
        """원근 변환 강도의 범위를 설정합니다."""
        self.perspective_transform_range = (max(0.0, min_value), min(1.0, max_value))
        # 현재 intensity가 새 범위를 벗어나면 조정
        self.perspective_intensity = max(min_value, min(max_value, self.perspective_intensity))

    def set_perspective_direction(self, direction):
        """원근 변환의 방향을 설정합니다."""
        valid_directions = [None, 'left', 'right', 'top', 'bottom', 'top_left', 'top_right', 'bottom_left', 'bottom_right']
        if direction in valid_directions:
            self.perspective_direction = direction
        else:
            raise ValueError("Invalid perspective direction")

config = TableGenerationConfig()

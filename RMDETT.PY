
# 2. 설정 파일 준비
from mmengine import Config
from mmdet.utils import register_all_modules

register_all_modules()

cfg = Config.fromfile('configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py')

# 필요한 경우 설정 수정
cfg.model.bbox_head.num_classes = 80  # COCO 데이터셋의 클래스 수
cfg.train_dataloader.batch_size = 8  # 배치 사이즈 조정

# 3. 학습
from mmdet.apis import train_detector
from mmdet.models import build_detector
from mmdet.datasets import build_dataset

# 모델 생성
model = build_detector(cfg.model)

# 데이터셋 생성
datasets = [build_dataset(cfg.train_dataloader.dataset)]

# 학습 실행
train_detector(model, datasets, cfg, distributed=False, validate=True)

# 4. 추론
from mmdet.apis import init_detector, inference_detector
import mmcv

# 모델 로드
checkpoint_file = 'work_dirs/rtmdet_tiny_8xb32-300e_coco/latest.pth'
model = init_detector(cfg, checkpoint_file, device='cuda:0')

# 이미지에 대한 추론 수행
img = mmcv.imread('test.jpg')
result = inference_detector(model, img)

# 결과 시각화
model.show_result(img, result, out_file='result.jpg')

# 5. 평가
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader

# 테스트 데이터셋과 데이터 로더 생성
dataset = build_dataset(cfg.test_dataloader.dataset)
data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=2)

# 평가 수행
results = single_gpu_test(model, data_loader)

# 평가 지표 계산
from mmdet.evaluation import eval_map
eval_map(results, dataset.data_infos, dataset.ann_file)
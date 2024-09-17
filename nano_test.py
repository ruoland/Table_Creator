import cv2
import torch
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config, load_model_weight
from nanodet.data.transform import Pipeline

# 설정 파일 로드
config_path = r"C:\Users\opron\Downloads\nanodet\config.yaml"
model_path = r"C:\Users\opron\Downloads\model_last.ckpt"
load_config(cfg, config_path)

model = build_model(cfg.model)
load_model_weight(model, model_path, None)

model.eval()

# 전처리 파이프라인 설정
pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

# 클래스 이름 리스트 (예시)
class_names = ["class1", "class2", "class3", "class4", "class5"]

def inference(img_path):
    img = cv2.imread(img_path)
    img_info = {"id": 0}
    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    meta = dict(img_info=img_info, raw_img=img, img=img)
    meta = pipeline(meta, cfg.data.val.input_size)
    meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).unsqueeze(0).cuda()

    with torch.no_grad():
        results = model.inference(meta)

    for det in results[0]:
        if det[4] > 0.3:  # 신뢰도 임계값
            x1, y1, x2, y2 = det[:4].astype(int)
            score = det[4]
            cls_id = int(det[5])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{class_names[cls_id]}: {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 테스트 실행
test_image_path = r"OCR\2_binary.jpg"
inference(test_image_path)

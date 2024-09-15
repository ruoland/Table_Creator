from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

def build_efficientdet(input_shape, num_classes):
    base_model = EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet')
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    output = Dense(num_classes * 4, activation='linear')(x)  # 4 = [x, y, width, height]
    
    model = keras.Model(inputs=base_model.input, outputs=output)
    return model

# 모델 생성 및 컴파일
input_shape = (512, 512, 3)  # 예시 입력 크기
num_classes = 5  # 테이블, 셀, 행, 열, 병합된 셀

model = build_efficientdet(input_shape, num_classes)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

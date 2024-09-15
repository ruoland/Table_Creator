# 데이터 전처리 함수
def preprocess_data(images, annotations, input_shape):
    processed_images = []
    processed_labels = []
    
    for img, ann in zip(images, annotations):
        resized_img = tf.image.resize(img, input_shape[:2])
        processed_images.append(resized_img)
        
        label = [0] * (num_classes * 4)
        label[ann['category_id']*4:(ann['category_id']+1)*4] = ann['bbox']
        processed_labels.append(label)
    
    return np.array(processed_images), np.array(processed_labels)

# 데이터 전처리
train_images_processed, train_labels = preprocess_data(train_images, train_annotations, input_shape)
val_images_processed, val_labels = preprocess_data(val_images, val_annotations, input_shape)

# 모델 학습
history = model.fit(
    train_images_processed, train_labels,
    validation_data=(val_images_processed, val_labels),
    epochs=50,
    batch_size=16
)

# 모델 저장
model.save('table_detection_model.h5')
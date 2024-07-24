# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("image-to-text", model="ddobokki/ko-trocr")

print(pipe('C:\\Users\\admin\\OneDrive\\OCR-PROJECT\\OCR\\OCR5-5.png'))
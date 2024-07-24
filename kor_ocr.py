# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("image-to-text", model="ddobokki/ko-trocr")

print(pipe('https://help.miricanvas.com/hc/article_attachments/900001490243/__________._8.png'))
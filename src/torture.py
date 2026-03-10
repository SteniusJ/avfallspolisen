# Test model

from ultralytics import YOLO

model = YOLO('./runs/output/train/weights/best.pt')

results = model.predict('../datasets/torture', save=True)
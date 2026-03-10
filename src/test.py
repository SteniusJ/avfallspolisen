# Test model

from ultralytics import YOLO

model = YOLO('./runs/output/train/weights/best.pt')

metrics = model.val(data='../datasets/mydataset.yml', split='test')
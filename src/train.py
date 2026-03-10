from ultralytics import YOLO
model = YOLO('yolov8n.pt')

results = model.train(data='../datasets/mydataset.yml',
                      epochs=200,
                      project='../output')
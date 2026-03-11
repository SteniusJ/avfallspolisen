import cv2
from ultralytics import YOLO
import numpy as np
import math

model = YOLO('./runs/output/train2/weights/best.pt')
names = model.names

video_path = 0

video_capture = cv2.VideoCapture(video_path)

if not video_capture.isOpened():
    print('Error opening video stream')
    exit

while video_capture.isOpened():
    status, frame = video_capture.read()

    if status:
        results = model(frame)

        for result in results:
            boxes = result.boxes.xyxy.numpy()
            class_names = []

            for c in result.boxes.cls:
                class_names.append(names[int(c)])
            
            print('class names:', class_names)

            for index, box in enumerate(boxes):    
                rwidth = box/frame.shape[1]

                #                     video   cordinater för rectangle                               färg brg   pixel bredd
                frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 20)    
                frame = cv2.putText(frame, class_names[index], (int(box[0]), int(box[1]) - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

        frame = cv2.resize(frame, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Avfallspolisen", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

video_capture.release()
cv2.destroyAllWindows()

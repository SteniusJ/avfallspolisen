import cv2
from ultralytics import YOLO
import numpy as np
import sys
import math

model = YOLO('./runs/output/train/weights/best.pt')
names = model.names

video_path = 0
if len(sys.argv) > 1:
    video_path = sys.argv[1]

confidence_threshold_mod = 0.2
dist_threshold = 200

video_capture = cv2.VideoCapture(video_path)

if not video_capture.isOpened():
    print('Error opening video stream')
    exit

def distance(point1, point2):
    return math.sqrt(((point2[0] - point1[0]) ** 2) + ((point2[1] - point1[1]) ** 2))

def analyze_frame(frame):
    results = model(frame)

    for result in results:
        class_names = []
        confidences = result.boxes.conf.numpy()
        confidence_threshold = np.average(confidences) - confidence_threshold_mod
        if np.isnan(confidence_threshold):
            confidence_threshold = 0

        for c in result.boxes.cls:
            class_names.append(names[int(c)])
        
        midpoints = []
        for box_loc in result.boxes.xyxy.numpy():
            midpoint = (int((box_loc[0] + box_loc[2]) / 2), int((box_loc[1] + box_loc[3]) / 2))
            midpoints.append(midpoint)

        for index, box in enumerate(result.boxes):
            self_midpoint = midpoints[index]

            if box.conf.item() < confidence_threshold:
                continue
            
            x1, y1, x2, y2 = box.xyxy[0]
            # bgr
            color = (0, 0, 255)

            for m_index, midpoint in enumerate(midpoints):
                if m_index == index:
                    continue
                if class_names[m_index] != class_names[index]:
                    continue

                if distance(self_midpoint, midpoint) < dist_threshold and confidences[m_index] > confidence_threshold:
                    color = (0, 255, 0)
                    break

            #                     video   cordinater för rectangle                      pixel bredd
            frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 5)    
            frame = cv2.putText(frame, class_names[index], (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            frame = cv2.putText(frame, str(round(box.conf.item(), 2)), (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            frame = cv2.putText(frame, "confidence threshold: " + str(round(confidence_threshold, 2)), (0, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

while video_capture.isOpened():
    status, frame = video_capture.read()

    if status:
        analyze_frame(frame)
        
        frame = cv2.resize(frame, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Avfallspolisen", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

video_capture.release()
cv2.destroyAllWindows()
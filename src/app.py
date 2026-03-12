import cv2
from ultralytics import YOLO
import numpy as np
import sys
import math

model = YOLO('./runs/output/train/weights/best.pt')
names = model.names

# video_path defaults to 0 (webcam view)
video_path = 0
# If program arguments are given it will be used to define video file path
if len(sys.argv) > 1:
    video_path = sys.argv[1]

# Thresholds
#
# Confidence threshold modifier: defines how many points the dynamic threshold should be reduced by
confidence_threshold_mod = 0.2
# Distance threshold: defines how close boxes have to be to be considered as "sorted"
dist_threshold = 200

video_capture = cv2.VideoCapture(video_path)

if not video_capture.isOpened():
    print('Error opening video stream')
    exit

def distance(point1, point2):
    """ Calculates distance between to points """
    # Function: distance = sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
    return math.sqrt(((point2[0] - point1[0]) ** 2) + ((point2[1] - point1[1]) ** 2))

def analyze_frame(frame):
    """ Analyzes video frame """
    results = model(frame)

    for result in results:
        class_names = []
        confidences = result.boxes.conf.numpy()

        # Dynamically calculates confidence_threshold
        # Confidence threshold defines how high the guess confidence has to be for the object to be included in calculations/draw
        confidence_threshold = np.average(confidences) - confidence_threshold_mod
        if np.isnan(confidence_threshold):
            confidence_threshold = 0

        for c in result.boxes.cls:
            class_names.append(names[int(c)])
        
        # Calculates box midpoints
        midpoints = []
        for box_loc in result.boxes.xyxy.numpy():
            # Calculate midpoint using the function: (xM, yM) = ((x1 + x2) / 2, (y1 + y2) / 2)
            midpoint = (int((box_loc[0] + box_loc[2]) / 2), int((box_loc[1] + box_loc[3]) / 2))
            midpoints.append(midpoint)

        for index, box in enumerate(result.boxes):
            self_midpoint = midpoints[index]
            x1, y1, x2, y2 = box.xyxy[0]
            # bgr (red)
            color = (0, 0, 255)

            # Check if box confidence is over threshold
            if box.conf.item() < confidence_threshold:
                continue

            for m_index, midpoint in enumerate(midpoints):
                # If self skip
                if m_index == index:
                    continue

                # If not same class skip
                if class_names[m_index] != class_names[index]:
                    continue

                # If distance below threshold and confidence below threshold change colour to green
                if distance(self_midpoint, midpoint) < dist_threshold and confidences[m_index] > confidence_threshold:
                    color = (0, 255, 0)
                    break

            # Draw rectangle
            frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 5)
            # Draw class name
            frame = cv2.putText(frame, class_names[index], (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            # Draw confidence
            frame = cv2.putText(frame, str(round(box.conf.item(), 2)), (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            # Draw confidence threshold
            frame = cv2.putText(frame, "confidence threshold: " + str(round(confidence_threshold, 2)), (0, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

# Main program loop
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

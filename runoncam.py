'''
Run model yolo on camera
'''

# Import libraries
import sys
import math
import time
import cv2
import psutil
from ultralytics import YOLO

# Load model
model = YOLO('best_ncnn_model')
model.imgsz = (360, 360)

# Set camera
URL = 'http://10.126.4.53:4747/video'
cap = cv2.VideoCapture(URL)
cap.set(3, 480)
cap.set(4, 480)

# Check camera
if not cap.isOpened():
    print('Error: Could not open camera.')
    sys.exit()
else:
    print('Success: Opened camera.')

# Object classes
classNames = ["Glass-Bottle", "Plastic-Bottle", "Trash"]

# Check bottle
def checkBottle(results):
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            confidence = math.ceil(box.conf[0] * 100) / 100
            if (classNames[cls] == 'Glass-Bottle' or classNames[cls] == 'Plastic-Bottle') and confidence > 0.8:
                return True
    return False
# Run model on camera
while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break
    # Run model
    results = model(frame)
    # Check bottle
    if checkBottle(results):
        print('Bottle detected!')
    else:
        print('No bottle detected!')
    # Font
    font = cv2.FONT_HERSHEY_SIMPLEX
    FONTSCALE = 0.5
    COLOR = (0, 255, 0)
    THICKNESS = 2
    # Draw bounding boxes and labels
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Print labels
            cls = int(box.cls[0])
            print("Class: ", classNames[cls])
            # Print confidence
            conf = math.ceil(box.conf[0] * 100) / 100
            print("Confidence: ", conf)
            # Object class
            org = [x1, y1]
            cv2.putText(frame, classNames[cls], org, font, FONTSCALE, COLOR, THICKNESS, cv2.LINE_AA)
             # Confidence
            org = [x1, y1 + 20]
            cv2.putText(frame, str(conf), org, font, FONTSCALE, COLOR, THICKNESS, cv2.LINE_AA)
    # Show FPS on screen after processing is done
    fps = cap.get(cv2.CAP_PROP_FPS)
    font = cv2.FONT_HERSHEY_SIMPLEX
    COLOR = (0, 255, 0)
    THICKNESS = 2
    FONTSCALE = 1
    cv2.putText(frame, 'FPS: ' + str(int(fps)), (10, 30), font, 1, COLOR, THICKNESS, cv2.LINE_AA)
    # Display result
    cv2.imshow('frame', frame)
    # Wait for 1.0 second
    time.sleep(1.0)
    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# CPU usage
print('Program takes CPU:', psutil.cpu_percent(), '%')
# Getting % usage of virtual_memory ( 3rd field)
print('RAM memory % used:', psutil.virtual_memory()[2])
# Getting usage of virtual_memory in GB ( 4th field)
print('RAM used (GB):', psutil.virtual_memory()[3]/1000000000)
# Release camera





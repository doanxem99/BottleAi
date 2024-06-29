'''
Script to run YOLOv8 model on the Picamera2
'''

import psutil
import cv2
import time
from picamera2 import Picamera2
import math
from ultralytics import YOLO

# Initialize the Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (380, 200)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load the YOLOv8 model
model = YOLO("best_ncnn_model", )

# Define the class names
classNames = ["Glass-Bottle", "Plastic-Bottle", "Trash"]


CONFIDENCE_THRESHOLD = 0.8
# Check bottle
def check_bottle(results):
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            confidence = math.ceil(box.conf[0] * 100) / 100
            if classNames[cls] == 'Plastic-Bottle' and confidence > CONFIDENCE_THRESHOLD:
                return True
    return False


# Main loop
while True:
    # Capture frame-by-frame
    frame = picam2.capture_array()

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Check if a bottle is detected
    if check_bottle(results):
        print("Bottle detected!")
    else:
        print("No bottle detected!")

    # Display the frame
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


    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

    # Pause for a while
    time.sleep(1.0)

# Release resources and close windows
cv2.destroyAllWindows()
picam2.stop()

# CPU usage
print('Program takes CPU:', psutil.cpu_percent(), '%')
# Getting % usage of virtual_memory ( 3rd field)
print('RAM memory % used:', psutil.virtual_memory()[2])
# Getting usage of virtual_memory in GB ( 4th field)
print('RAM used (GB):', psutil.virtual_memory()[3]/1000000000)

'''
Script to run YOLOv8 model on the Picamera2
'''

import psutil
import cv2
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
model = YOLO("best_ncnn_model")

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

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release resources and close windows
cv2.destroyAllWindows()


# CPU usage
print('Program takes CPU:', psutil.cpu_percent(), '%')
# Getting % usage of virtual_memory ( 3rd field)
print('RAM memory % used:', psutil.virtual_memory()[2])
# Getting usage of virtual_memory in GB ( 4th field)
print('RAM used (GB):', psutil.virtual_memory()[3]/1000000000)

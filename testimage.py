from ultralytics import YOLO
import cv2
import psutil
import time

start = time.time()

# Load model
model = YOLO('best_ncnn_model', task='detect')

# Test on a single image
results = model('test.jpg')
for result in results:
    img = result.plot()
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print('Time taken:', time.time()-start)

# CPU usage
print('Program takes CPU:', psutil.cpu_percent(), '%')
# Getting % usage of virtual_memory ( 3rd field)
print('RAM memory % used:', psutil.virtual_memory()[2])
# Getting usage of virtual_memory in GB ( 4th field)
print('RAM used (GB):', psutil.virtual_memory()[3]/1000000000)

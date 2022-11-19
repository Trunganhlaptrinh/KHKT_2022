import cv2
import torch

#%matplotlib inline sua loi --> khong can thiet --> dong nay chi dung voi Jupyter
from matplotlib import pyplot as plt
import numpy as np
import cv2
import cv2 as cv
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
vidcap = cv2.VideoCapture(0)
success, image = vidcap.read()
count = 0
import cv2

# Define a video capture object
vidcap = cv2.VideoCapture(0)

# Capture video frame by frame
success, image = vidcap.read()

# Declare the variable with value 0
count = 0

# Creating a loop for running the video
# and saving all the frames
while success:

	# Capture video frame by frame
	success, image = vidcap.read()

	# Resize the image frames
	resize = cv2.resize(image, (700, 500))

	# Saving the frames with certain names
	cv2.imwrite("%04d.jpg" % count, resize)

	# Closing the video by Escape button
	if cv2.waitKey(10) == 27:
		break

	# Incrementing the variable value by 1
	count += 1


while vidcap.isOpened():
    ret, frame = vidcap.read()
    
    # Make detections 
    results = model(frame)
    
    cv2.imshow('KHKT', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
vidcap.release()
cv2.destroyAllWindows()



import torch

#%matplotlib inline sua loi --> khong can thiet --> dong nay chi dung voi Jupyter
from matplotlib import pyplot as plt
import numpy as np
import cv2
import cv2 as cv
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


# img = 'https://www.google.com/search?q=traffic+jam&source=lnms&tbm=isch&sa=X&ved=2ahUKEwib8qTP07n7AhWWFIgKHeshA9kQ_AUoAXoECAIQAw&biw=1536&bih=696&dpr=1.25#imgrc=byVg5IKc-ip2dM'
# results = model(img)
# results.print()

# plt.imshow(np.squeeze(results.render()))
# plt.show()

# results.render()

# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()
    
#     # Make detections 
#     results = model(frame)
    
#     cv2.imshow('KHKT', np.squeeze(results.render()))
    
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

import uuid  
import os
import time


IMAGES_PATH = os.path.join('save_img', 'img') #luu duong dan -- os -- path
labels = ['LAM_BAI', 'GIAN_LAN'] # lỗi đoạn này,  không nên để tiếng việt
number_imgs = 30 # số lượng ảnh  <=> số lần lặp

cap = cv2.VideoCapture(0)
# Loop through labels
for label in labels:
    print('Collecting images for {}'.format(label))
    time.sleep(7) # thời gian chuyển các ảnh 
    
    # Loop through image range
    for img_num in range(number_imgs):
        print('Collecting images for {}, image number {}'.format(label, img_num))
        
        # Webcam feed
        ret, frame = cap.read()
        
        # Naming out image path
        imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
        
        # Writes out image to file 
        cv2.imwrite(imgname, frame)
        
        # Render to the screen
        cv2.imshow('Image Collection', frame)
        
        # 2 second delay between captures
        time.sleep(2)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

print(os.path.join(IMAGES_PATH, labels[0]+'.'+str(uuid.uuid1())+'.jpg'))

for label in labels:
    print('Collecting images for {}'.format(label))
    for img_num in range(number_imgs):
        print('Collecting images for {}, image number {}'.format(label, img_num))
        imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
        print(imgname) 
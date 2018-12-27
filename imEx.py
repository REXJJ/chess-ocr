import cv2
import numpy as np
from scipy import ndimage, misc
from matplotlib import pyplot as plt
import time

img = cv2.imread('chee.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = img.astype('uint8')
#retval, th = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
#img = cv2.Laplacian(img,cv2.CV_64F)



height, width = img.shape
CROP_W_SIZE  = 8
CROP_H_SIZE = 8
count=0
for ih in range(CROP_W_SIZE):
  for iw in range(CROP_H_SIZE):
      count=count+1
      if count==4:
        x = int(width/CROP_W_SIZE * iw) 
        y = int(height/CROP_H_SIZE * ih)
        h = int(height / CROP_H_SIZE)
        w = int(width / CROP_W_SIZE )
        #print(x,y,h,w)
        img = img[y+int(h/10):y+int(h)-int(h/10), x+int(w/10):x+int(w)-int(w/10)]
        NAME = "king"+str(count)
        cv2.imwrite("./Crop/" + NAME +  ".png",img)
        time.sleep(5) 

      if count==60:
        x = int(width/CROP_W_SIZE * iw) 
        y = int(height/CROP_H_SIZE * ih)
        h = int(height / CROP_H_SIZE)
        w = int(width / CROP_W_SIZE )
        #print(x,y,h,w)
        img = img[y+int(h/10):y+int(h)-int(h/10), x+int(w/10):x+int(w)-int(w/10)]
        NAME = "king"+str(count)
        cv2.imwrite("./Crop/" + NAME +  ".png",img)
        time.sleep(5) 

    


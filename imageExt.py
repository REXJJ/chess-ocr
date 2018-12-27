import cv2
import numpy as np
from scipy import ndimage, misc
from matplotlib import pyplot as plt
import time

import os
filelist=os.listdir('./')
kernel = np.ones((1,1),np.uint8)
cou=0
maximum=0
for fichier in filelist[:]:
    if (fichier.endswith(".png")):
        print fichier
        img = cv2.imread(fichier)
        #print img.shape
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = img.astype('uint8')
        #retval, img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
        #img = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        #img = cv2.erode(th,kernel,iterations = 1)
        #img = cv2.Laplacian(img,cv2.CV_64F)
        #img=255-img
        height, width = img.shape
        CROP_W_SIZE  = 8
        CROP_H_SIZE = 8
        count=0
        cou=cou+1
        for ih in range(CROP_W_SIZE):
          for iw in range(CROP_H_SIZE):
               count=count+1
               summ=0
               if count == 64:
                   print('Done')
               
               img_copy=img
               x = int(width/CROP_W_SIZE * iw) 
               y = int(height/CROP_H_SIZE * ih)
               h = int(height / CROP_H_SIZE)
               w = int(width / CROP_W_SIZE )
               #print(x,y,h,w)
               img = img[y+int(h/10):y+int(h)-int(h/10), x+int(w/10):x+int(w)-int(w/10)]
               #retval, img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
        
               '''xx,yy=img.shape
               f=2.4
               #print (xx,yy)
               ct=0
               center = img[int(yy/f):yy-int(yy/f),int(xx/f):xx-int(xx/f)]
               for i in center:
                   for j in i:
                       summ=summ+j
                       ct=ct+1
               summ=summ/ct
               ct=0'''
               '''_, contours, _= cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
               cnt = max(contours, key = cv2.contourArea)
               x, y, w, h = cv2.boundingRect(cnt)
               roi = img [y:y+h, x:x+w]
               summ=0
               coun=0
               for x in roi:
                  for y in x:
                     summ=summ+y
                     coun=coun+1
               print summ/coun

               if maximum<(summ/coun):
                   maximum=summ/coun'''
               NAME = "blankc"+str(count)+str(cou)
               #if (count>=0 and count<17) or (count>48 and count<65):
               if(count>16 and count<49):
                  cv2.imwrite("./Spaces/" + NAME +  ".png",img)
            
               
               img=img_copy
        #print(maximum)
    

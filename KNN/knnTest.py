import cv2
from PIL import Image
import dataExt

model1=dataExt.train('./colorsamp.data','./colorrespo.data')
model2=dataExt.train('./piecessamp.data','./piecesrespo.data')
model3=dataExt.train('./psamp.data','./prespo.data')

img=cv2.imread('./chesst.png')               
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = img.astype('uint8')
retval, th = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
#img = cv2.Laplacian(img,cv2.CV_64F)
#img=255-img
height, width = img.shape
CROP_W_SIZE  = 8
CROP_H_SIZE = 8
count=0
storage=[]
arr=['k','q','r','b','n','p','v']
sq_len = int(img.shape[0] / 8)
for i in range(8):
        board_s=''
        for j in range(8):
               img_copy=img
               img=(img[i * sq_len : (i + 1) * sq_len, j * sq_len : (j + 1) * sq_len])
               s1='./ches.png'
               cv2.imwrite(s1,img)

               if dataExt.predictor(s1,model3):
                 t=dataExt.predictor(s1,model2)
                 t=arr[t]
                 print t
                 alp=dataExt.predictor(s1,model1)
                 #print (alp)
                 if alp==0:
                     t=t.upper()
                 board_s=board_s+t
               else:
                 board_s=board_s+'-'
                 
                  
               img=img_copy
        
        storage.append(board_s)
for x in storage:
  print(x)

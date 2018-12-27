import cv2
import numpy as np
from scipy.interpolate import griddata
import time
import os
from sklearn import svm

def predictor(im,model):
    im=cv2.imread(im)
    out = np.zeros(im.shape,np.uint8)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
    image, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
      if cv2.contourArea(cnt)>50:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h>28:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            roismall = roismall.reshape((1,100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
            #return model.predict(roismall)
            string = str(int((results[0][0])))
            return int(results[0][0])
    return 2


'''
def update(sudoku):
    samples = np.loadtxt('generalsamples.data',np.float32)
    responses = np.loadtxt('generalresponses.data',np.float32)
    for i in range(9):
        for j in range(9):
            file='./CROP/square'+str(i)+str(j)+'.png'
            im=cv2.imread(file)
            out = np.zeros(im.shape,np.uint8)
            gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
            image, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
              if cv2.contourArea(cnt)>50:
                [x,y,w,h] = cv2.boundingRect(cnt)
                if  h>28:
                    cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
                    roi = thresh[y:y+h,x:x+w]
                    roismall = cv2.resize(roi,(10,10))
                    roismall = roismall.reshape((1,100))
                    #roismall = np.float32(roismall)
                    if sudoku[i][j]!=0:
                        samples=np.append(samples,roismall,0)
                        responses=np.append(responses,np.array(float(sudoku[i][j])))

    

    np.savetxt('generalsamples.data',samples)
    np.savetxt('generalresponses.data',responses)    
'''    
'''    
samples =  np.empty((0,100))
responses = []
samples = np.loadtxt('psamp.data',np.float32)
responses = np.loadtxt('prespo.data',np.float32)
files='./Pieces/'    
filelist=os.listdir(files)
for fichier in filelist[:]:
    if (fichier.endswith(".png")):
            print fichier
            im=cv2.imread(files+fichier)
            out = np.zeros(im.shape,np.uint8)
            gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
            image, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
              if cv2.contourArea(cnt)>50:
                [x,y,w,h] = cv2.boundingRect(cnt)
                if  h>28:
                    cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
                    roi = thresh[y:y+h,x:x+w]
                    roismall = cv2.resize(roi,(10,10))
                    roismall = roismall.reshape((1,100))
                    #roismall = np.float32(roismall)
                    samples=np.append(samples,roismall,0)
                    responses=np.append(responses,np.array(int(1)))

    

np.savetxt('psamp.data',samples)
np.savetxt('prespo.data',responses)    
'''    
def train(samples,responses):
    samples = np.loadtxt(samples,np.float32)
    responses = np.loadtxt(responses,np.float32)
    responses = responses.reshape((responses.size,1))
    #clf = svm.SVC()
    #clf.fit(samples, responses)

    
    model = cv2.ml.KNearest_create()
    model.train(samples,cv2.ml.ROW_SAMPLE,responses)
    return model
   
'''model=train()
image='./whiter.png'
pred=predictor(image,model)
print pred
'''

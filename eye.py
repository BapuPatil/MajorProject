import numpy as np
import cv2 as cv
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
mouth_cascade = cv.CascadeClassifier('haarcascade_mcs_mouth.xml') 
img=cv.imread('01.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    mouth = mouth_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    for(mx,my,mw,mh) in mouth:
        cv.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,0,255),2)
        break
    
cv.imwrite('fac.jpg',img)
cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()
    

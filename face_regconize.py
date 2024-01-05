import numpy as np 
import cv2 as cv 

haar_cascade = cv.CascadeClassifier('D:\OPEN CV\Face Detect and Regconize\haar_face.xml')

people = ["Bill Gates", "Elon Musk"]
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)

face_regconizer = cv.face.LBPHFaceRecognizer.create()
face_regconizer.read('face_trained.yml')

img = cv.imread('Resources/Test file/Bill.jfif')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

#Detect the face in the image

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    label, confidence = face_regconizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')
    print(label)
    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.putText(img, str(confidence), (20,50), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

cv.imshow('Detected', img)
cv.waitKey(0)
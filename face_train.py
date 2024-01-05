import os
import cv2 as cv 
import numpy as np 
from matplotlib import pyplot as plt

people = []
path = "D:\OPEN CV\Face Detect and Regconize\Resources"
for i in os.listdir(path):
    people.append(i)
# print(people)


haar_cascade = cv.CascadeClassifier('D:\OPEN CV\Face Detect and Regconize\haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        p_path = os.path.join(path, person)
        label = people.index(person)
        # print(label)
        for img in os.listdir(p_path):
            img_path = os.path.join(p_path,img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            # cv.imshow(str(img_path),img_array)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                face_roi = gray[y:y+h, x:x+w]
                features.append(face_roi)
                labels.append(label)

create_train()
print('Training done')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_regconizer = cv.face.LBPHFaceRecognizer.create()

# Train regconizer on features and labels list
face_regconizer.train(features,labels)

face_regconizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)


cv.waitKey(0)
            

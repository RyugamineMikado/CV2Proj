import cv2
import numpy as np
import pickle
import keras 
from keras.lay

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
#Face Recognizer implementare 
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml") #Fisier generat pe baza a faces-train.py ce contine modelul ML

labels = {}
with open("labels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}


#Use video instead of camera feed
vid = cv2.VideoCapture(r'D:\Python Proj\test.mp4')

while(vid.isOpened() ):
    #Capture frame by frame 
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors= 5)

    #Display location for face 
    for (x, y, w, h ) in faces:
        #print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        #Recognize dl model, predict face
        id_,conf = recognizer.predict(roi_gray) 
        #print(id_, conf)

        if conf>=45 and conf <=100:
            print(labels[id_])

        #Rectangle details
        color = (255, 0, 0) #BGR
        stroke = 3 
        end_cord_x= x + w 
        end_cord_y= y + h
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y) ,color,stroke)

    if ret ==True:
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

vid.release()
cv2.destroyAllWindows()

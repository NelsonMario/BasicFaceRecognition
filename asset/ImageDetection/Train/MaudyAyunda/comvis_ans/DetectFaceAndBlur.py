import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math

train_path = 'asset\ImageDetection\Train'
person_name = os.listdir(train_path)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_list = []
class_list = []

for idx, name in enumerate(person_name):
    person_path = train_path + '/' + name
    for image_name in os.listdir(person_path):
        full_path = person_path + '/' + image_name
        img = cv2.imread(full_path, 0)

        detected_face = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=10)

        if(len(detected_face) < 1):
            continue

        for face_rect in detected_face:
            x, y, h, w = face_rect
            face_img = img[y:y+h, x:x+w]
            
            face_list.append(face_img)
            class_list.append(idx)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(face_list, np.array(class_list))

test_path = 'asset\ImageDetection\Test'

for img in os.listdir(test_path):
    full_path = test_path + '/' + img
    full_color_image = cv2.imread(full_path)
    img = cv2.imread(full_path, 0)
    
    detected_face = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

    

    if(len(detected_face) < 1):
        continue

    for face_rect in detected_face:
        x, y, h, w = face_rect
        face_img = full_color_image[y:y+h, x:x+w]
        
  
        blur = cv2.GaussianBlur(face_img, (101, 101), 0)
        
        full_color_image[y:y+h, x:x+w] = blur

        cv2.imshow("Blur", full_color_image)
        cv2.waitKey(0)
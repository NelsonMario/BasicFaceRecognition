import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math
import pickle


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

filename = 'face_recog.clf'
pickle.dump(face_recognizer, open(filename, 'rb'))

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
        face_img = img[y:y+h, x:x+w]
        
        result, confidence = face_recognizer.predict(face_img)
        confidence = math.floor(confidence * 100) / 100
        plt.title(person_name[result] + ":" + str(confidence) + "%")
        full_color_image = cv2.cvtColor(full_color_image, cv2.COLOR_BGR2RGB)
        cv2.rectangle(full_color_image, (x, y), (x + w, y + h), [0, 255, 0], 1)
        plt.imshow(full_color_image)
        plt.show()
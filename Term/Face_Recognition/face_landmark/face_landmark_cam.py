import dlib
import cv2
import os
import numpy as np

ALL = list(range(0,68))
JAWLINE = list(range(0,17))
RIGHT_EYEBROW = list(range(17,22))
LEST_EYEBROW = list(range(22,27))
NOSE = list(range(27,36))
RIGHT_EYE = list(range(36,42))
LEFT_EYE = list(range(42,48))
MOUTH_OUTLINE = list(range(48,61))
MOUTH_INNER = list(range(61,68))

face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

def empty(a):
    pass

state = 0
cnt = 0

base_dir = 'C:/Users/Hong/Desktop/term_project/'
cap_dir = os.path.join(base_dir)

while True:    
    success, img = cap.read()
    
    image = cv2.resize(img, dsize=(640,480), interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

    faces = face_detector(img)
    
    if len(faces) >= 1 :
        state = 1
        
        for f in faces:
            horizon = f.right()-f.left()
            vertical = f.bottom()-f.top()
            size = horizon * vertical
            
            if size > 45000 :
                print(f"face are detected. {cnt}")
                cv2.rectangle(img, (f.left(), f.top()), (f.right(), f.bottom()), (0,0,255),2)
                
                landmarks = predictor(image, f)
                landmark_list = []
                
                for p in landmarks.parts():
                    landmark_list.append([p.x, p.y])
                    cv2.circle(image, (p.x, p.y), 2, (0,255,0), -1)
                
                
                crop = img[f.top():f.bottom(), f.left():f.right()]
                saved_img_path = os.path.join(cap_dir, f'Hong.jpg')
                cv2.imwrite(saved_img_path, crop)
                cnt+=1
            
    elif len(faces) == 0:
        cnt = 0
        state = 0
            
    cv2.imshow("Cam", image)
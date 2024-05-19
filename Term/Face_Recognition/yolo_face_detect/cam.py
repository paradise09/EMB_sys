import cv2
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

'''frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)'''

video_path = 0
min_confidence = 0.5

def detectAndDisplay(frame):
    start_time = time.time()
    img = cv2.resize(frame, None, fx=0.8, fy=0.8)
    height, width, channels = img.shape
    
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > min_confidence:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    font = cv2.FONT_HERSHEY_DUPLEX
    
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = "{}: {: .2f}".format(classes[class_ids[i]],confidences[i]*100)
            print(i, label)
            color = colors[i]
            
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img,label, (x,y-5), font, 1, color, 1)
    end_time = time.time()
    process_time = end_time = start_time
    print("===A frame took {: .3f} seconds".format(process_time))
    cv2.imshow("screen", img)
            
model_file = 'face_classification.h5'
config_file = 'yolov3-320.cfg'
net = cv2.dnn.readNet(model_file, config_file)

classes = []
with open("person.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,siz=(len(classes),3))

def empty(a):
    pass

'''base_dir = 'C:/Users/Hong/Desktop/term_project/'
cap_dir = os.path.join(base_dir, 'predict')

model = load_model('C:/Users/Hong/Desktop/term_project/face_classification.h5')

class_names = {0 : 'Choi', 1 : 'Hong', 2 : 'Kang'}
'''
cap = cv2.VideoCapture(video_path)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    success, frame = cap.read()
    if frame is None:
        print('--(!)No captured frame --Break!')
        break
    detectAndDisplay(frame)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cv2.detroyAllWindows()
    
    
'''elif key & 0xFF == ord('s'):
        saved_img_path = os.path.join(cap_dir, 'predict.jpg')
        cv2.imwrite(saved_img_path, img)
        print("Screen shot!")
        
        # image preprocessing
        img_to_predict = image.load_img(saved_img_path, color_mode='grayscale', target_size=(150,150))
        img_to_predict = image.img_to_array(img_to_predict)
        img_to_predict = np.expand_dims(img_to_predict, axis=0)
        img_to_predict /= 255.0
        
        output = model.predict(img_to_predict)
        predicted = np.argmax(output, axis=1)
        predicted_label = class_names[predicted[0]]
        confidence = output[0][predicted[0]]*100
        
        print(f"Predicted Label : {predicted_label}, Confidence : {confidence}")'''
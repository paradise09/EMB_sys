import os
import cv2
import numpy as np

from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC

data = []
labels = []

base_dir = 'C:/Users/Hong/Desktop/Face_Recognition/face_data/train/'

class_names = ['Kang', 'Hong', 'Choi', 'Unknown']
label_dict = {name: i for i, name in enumerate(class_names)}

img_height, img_width = 224, 224

for class_name in class_names:
    dir_path = os.path.join(base_dir, class_name)
    
    for img_name in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_name)
        
        img = image.load_img(img_path, target_size=(img_height, img_width))
        img = image.img_to_array(img)
        img /= 255.0
        
        data.append(img)
        labels.append(label_dict[class_name])
        
data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))

pca = PCA(n_components=150, whiten=True, random_state=42)
X_train_pca = pca.fit_transform(X_train_flat)
X_test_pca = pca.transform(X_test_flat)

svc = SVC(kernel='rbf', class_weight='balanced')
svc.fit(X_train_pca, y_train)

#y_pred = svc.predict(X_test_pca)
#print(classification_report(y_test, y_pred))
#print("Accuracy: ", accuracy_score(y_test, y_pred))

def preprocess_image(image_path, target_size):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img,target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

image_path = '.jpg'

preprocessed_image = preprocess_image(image_path, (224,224))

flat_image = preprocessed_image.reshape((1,-1))
pca_image = pca.transform(flat_image)

predicted_class = svc.predict(pca_image)
print("Predicted Class:", predicted_class)
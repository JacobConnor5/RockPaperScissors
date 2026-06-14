import pandas as pd
import os
import sklearn,pickle
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
# STEP 1: Import the necessary modules.
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import main

print("testing")


mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles



Categories=['Rock','Paper','Scissors']
dataArr=[] #input array will have colour of each pixelin the array
target_arr=[] #output array will have the answer so thsese colours mean rock
datadir=''
#path which contains all the categories of images


for i in Categories:
    
    print(f'loading... category : {i}')
    path=os.path.join(datadir,i)
    print("length, ",len(os.listdir(path)))
    for img in os.listdir(path):
        print("path: ",os.path.join(path,img))
        image = mp.Image.create_from_file(os.path.join(path,img))
        #img_resized=resize(img_array,(150,150,3))

        detection_result = main.detector.detect(image)
        #print(annotated_image)
        #Fall back to the raw image if no landmarks detected
        print("test",os.path.join(path,img))
        raw_numpy = image.numpy_view().copy()
        
        if detection_result.hand_landmarks:  # or face_landmarks / pose_landmarks depending on your model
            annotated_image = main.draw_landmarks_on_image(raw_numpy, detection_result)
        else:
            annotated_image = raw_numpy  # use original if nothing detected
        
        
        resizedImg = cv2.resize(annotated_image, (150, 150))
        dataArr.append(resizedImg) 

        # dataArr.append(annotated_image.tolist())
        target_arr.append(Categories.index(i))
    print(f'loaded category:{i} successfully')

print(type(dataArr))
feat = np.array(dataArr)
feat = feat.reshape(feat.shape[0], -1)
label=np.array(target_arr)
print(label)

feat_train,feat_test,label_train,label_test = sklearn.model_selection.train_test_split(feat,label,test_size = 0.1)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(feat_train,label_train)
acc = model.score(feat_test,label_test)
predictions = model.predict(feat_test)

for i in range(len(predictions)):
    print('prediction:',predictions[i],'actual:',label_test[i])

print(acc)

with open('Rock.pickle','wb') as pickle_file:
    pickle.dump(model,pickle_file)
    pickle_file.close()


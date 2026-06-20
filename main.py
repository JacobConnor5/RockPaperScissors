import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pickle
import time
import random

def saved_model():
    temp = open('Rock.pickle','rb')
    modelData = pickle.load(temp)
    model = modelData['model']
    scaler = modelData['scaler']
    return model,scaler

def draw_landmarks_on_image(rgb_image, detection_result,prediction):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        xCoordinates = [landmark.x for landmark in hand_landmarks]
        yCoordinates = [landmark.y for landmark in hand_landmarks]

        minX = max(int(min(xCoordinates)*width),0)
        minY = max(int(min(yCoordinates)*height),0)
        maxX = min(int(max(xCoordinates)*width),1280)
        maxY = min(int(max(yCoordinates)*height),720)

        text_x = int(min(xCoordinates) * width)
        text_y = int(min(yCoordinates) * height) - MARGIN
        #print("text x: ",text_x)
        # Draw handedness (left or right hand) on the image.
        # cv2.putText(annotated_image, f"{handedness[0].category_name}",
        #             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
        #             FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        cv2.rectangle(annotated_image,(minX,minY),(maxX,maxY),(0,255,0),2)
        cv2.putText(annotated_image,CATEGORIES[prediction[0]],(text_x,text_y),cv2.FONT_HERSHEY_SIMPLEX,FONT_SIZE,HANDEDNESS_TEXT_COLOR,FONT_THICKNESS,cv2.LINE_AA)

        annotated_image = annotated_image
    return annotated_image

def imageProccesing(img):
    if len(img.shape) == 3:
        flat = img.flatten()

    else:
        flat = img.flatten()

    ready = np.expand_dims(flat,axis = 0)
    return ready

def play(prediction):
    choice = random.randint(0,2)
    time.sleep(1)
    print('You chose',CATEGORIES[prediction[0]])
    time.sleep(1)
    print('I chose',CATEGORIES[choice])

    if choice == prediction+1 or (choice==0 and prediction==2):
       print('You lose')
       score[1][1] += 1

    elif prediction == choice+1 or (prediction == 0 and choice == 2):
       print('Well done',name, 'you won')
       score[0][1] += 1

    start_time =time.time()

    return start_time

def main():
    model,scaler = saved_model()
    prediction = [0]
    start = time.time()
    current = time.time()
    count = ['ROCK','PAPER','SCISSORS','GO','']

    while True:
        attempt = 0 #if camera takes too long it will fail so attempt adds a failsafe 
        success,img = cap.read()
        while not success and attempt<5:
            time.sleep(0.5)
            success,img = cap.read()
            attempt+=1
        if not success:
            print("failed to read frame")
            break
        
        current = time.time()
        difference = round(current - start)

        if difference <0 or difference>4:
            difference=0

        img = cv2.flip(img,1)
        rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detectionResults = detector.detect(mp_image)

        if detectionResults.hand_landmarks:
            handLandmarks = detectionResults.hand_landmarks[0]
            landmarks = []
            for landmark in handLandmarks: 
                landmarks.extend([landmark.x,landmark.y,landmark.z])
            
            if landmarks:
                data = scaler.transform(np.array(landmarks).reshape(1,-1))
                prediction = model.predict(data)

        annotatedImage = draw_landmarks_on_image(mp_image.numpy_view(),detectionResults,prediction)
        backToOrig = cv2.cvtColor(annotatedImage,cv2.COLOR_RGB2BGR)

        #print("We Predict: ",CATEGORIES[prediction[0]])
        cv2.putText(backToOrig,count[difference],(0,360),cv2.FONT_HERSHEY_SIMPLEX,FONT_SIZE,HANDEDNESS_TEXT_COLOR,FONT_THICKNESS,cv2.LINE_AA)

        cv2.imshow("Image",backToOrig)

        if difference == 4:
            start = play(prediction)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

name = input('what is your name')
score = [[name,0],['CPU',0]]

CATEGORIES = ["Rock","Paper","Scissors"]
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)


cap = cv2.VideoCapture(0)

cap.set(3,1280)
cap.set(4,720) #setting the size of the capture 3 is the index for width and 4 is the index for height


if __name__=="__main__":
    main()
    
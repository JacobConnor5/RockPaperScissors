import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pickle
import time

def saved_model():
	temp = open('Rock.pickle','rb')
	model = pickle.load(temp)
	
	return model

def draw_landmarks_on_image(rgb_image, detection_result):
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

        annotated_image = annotated_image[minY:maxY,minX:maxX]

    return annotated_image

def imageProccesing(img):
    if len(img.shape) == 3:
        flat = img.flatten()

    else:
        flat = img.flatten()

    ready = np.expand_dims(flat,axis = 0)
    return ready


def main():
    model = saved_model()

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

        img = cv2.flip(img,1)
        rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detectionResults = detector.detect(mp_image)

        annotatedImage = draw_landmarks_on_image(mp_image.numpy_view(),detectionResults)
        resized = cv2.resize(annotatedImage,(150,150))
        resized = np.array(resized)
  
        resized = resized.reshape(resized.shape[0],-1)
        resized = imageProccesing(resized)
        #clean this the fuck up

        prediction = model.predict(resized)
        print("We Predict: ",CATEGORIES[prediction[0]])

        cv2.imshow("Image",annotatedImage)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

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
    
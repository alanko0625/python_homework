import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mphands = mp.solutions.hands
hands = mphands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    ret,img = cap.read()
    if ret:

        imgRGB = cv2.cvtColor(img ,cv2.COLOR_RGBA2RGB)
        result = hands.process(imgRGB)
        #print(result.multi_hand_landmarks)
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms,mphands.HAND_CONNECTIONS)

        cv2.imshow('img',img)
    if cv2.waitKey(1) == ord('q'):
        break

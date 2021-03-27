import cv2
import os
import time
import uuid
import pathlib
import mediapipe as mp

# Create object for mediapipe hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils  # used for drawing the hand landmarks

IMAGES_PATH = 'Tensorflow\workspace\images\collectedImages'

labels = ['hello', 'thanks', 'yes', 'no', 'iloveyou']
number_imgs = 15 # number of images to collect

for label in labels:
    # create directory
    #     !mkdir{'Tensorflow\\workspace\\images\\collectedImages\\'+label}
    os.makedirs(os.path.join(pathlib.Path().parent.absolute(), IMAGES_PATH, label))

    # open webCam
    cap = cv2.VideoCapture(0)

    print('Collecting images for {}'.format(label))
    time.sleep(5)

    for imgNum in range(number_imgs):
        ret, frame = cap.read()

        # convert img to rgb
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # process the img with mediapipe
        results = hands.process(imgRGB)

        # ensure system detect hand
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

        # imgName = os.path.join(os.path.join(pathlib.Path().parent.absolute(), IMAGES_PATH, label,
        #                                     label + '.' + '{}.jpg'.format(str(uuid.uuid1()))))
        # cv2.imwrite(imgName, frame)
        cv2.imshow('Frame', frame)
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
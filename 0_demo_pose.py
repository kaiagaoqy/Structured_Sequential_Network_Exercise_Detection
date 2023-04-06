
from utils_pose import *

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('test_2/MP_Data') 

# Actions that we try to detect
actions = np.array(['stride', 'squat', 'up', 'down'])
CATE = 4
# Thirty videos worth of data
no_sequences = 25
stride =1
# Videos are going to be 30 frames in length
sequence_length = 1


#plt.figure(figsize=(18,18))
#plt.imshow(prob_viz(res, actions, image, colors))

#sequence.reverse()


# 1. New detection variables
sequence = []
sentence = []
end=[]
starting=[]

threshold = 0.98
end_threshhold = 0.3
start_threshhold = 0.8

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(no_sequences,132)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.load_weights('/home/sstc/文档/action_detection/test_2/action_pose_mean_adam_50.h5')

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('/home/sstc/文档/action_detection/test_2/广播体操动作/VID20220328180027.mp4')
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        #print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)

        sequence.append(keypoints)
        sequence = sequence[-no_sequences-int(no_sequences/2):]
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == no_sequences:
            a = np.expand_dims(sequence, axis=0)
            last_end =  a[:,list(range(0,no_sequences,stride)),:]
            last_end = model.predict(last_end)[0]
            res = last_end
            pos = np.argmax(res)
        elif len(sequence) == no_sequences+int(no_sequences/2):
            
            a = np.expand_dims(sequence, axis=0)
            
            last_end =  a[:,list(range(0,no_sequences,stride)),:]
            course = a[:,list(range(int(no_sequences/2),no_sequences+int(no_sequences/2),stride)),:]

            last_end = model.predict(last_end)[0]
            res = model.predict(course)[0]
            pos = np.argmax(res)
            '''
            if res[np.argmax(res)] > threshold: 
                print(actions[np.argmax(res)],"---",res[np.argmax(res)])
            '''
            
        #3. Viz logic
            '''
            预测概率>threshold, 第一个&  输出max分类

            仿照SSN，设置starting，course，end
            '''
        if (cap.get(cv2.CAP_PROP_POS_FRAMES) == no_sequences) or (len(sequence) == no_sequences+int(no_sequences/2)):    
            if (res[pos] > threshold) & (last_end[pos]>start_threshhold): 
                if len(sentence) > 0: 
                    if (actions[np.argmax(res)] != sentence[-1]) & (last_end[last_max]< end_threshhold):
                        sentence.append(actions[np.argmax(res)])
                        last_max =  np.argmax(res)
                else:
                    sentence.append(actions[np.argmax(res)])
                    last_max = np.argmax(res)

            if len(sentence) > CATE+1: 
                #sentence = sentence[-CATE:]
                sentence = []

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


from matplotlib.contour import ContourSet
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
parts = 5
no_sequences = 25
stride =1
# Videos are going to be 30 frames in length
sequence_length = 1
def ret_part_ft(n_part,span,ft):
    stage_ft = ft[span[0]:span[1],:]
    duration = stage_ft.shape[0]
    ticks = range(0,duration+1000,int(duration/n_part))
    for i in range (n_part):
        if i == 0:
            part_ft = ft[ticks[i]:ticks[i+1],:].mean(axis=0)
        else:
            part_ft = np.vstack([part_ft,ft[ticks[i]:ticks[i+1],:].mean(axis=0)])
    return part_ft

#plt.figure(figsize=(18,18))
#plt.imshow(prob_viz(res, actions, image, colors))

#sequence.reverse()


# 1. New detection variables
sequence = []
sentence = []
end=[]
starting=[]
course=(1,2) ## 两层金字塔


threshold = 0.98
end_threshhold = 0.3
start_threshhold = 0.8

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(parts,132)))
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
        sequence = sequence[-no_sequences:]
        if cap.get(cv2.CAP_PROP_POS_FRAMES) >= no_sequences:
            a = np.expand_dims(sequence, axis=0)
            ft = a[0]
            n_parts = 2+course[1]
            ticks = range(0,no_sequences,int(no_sequences/n_parts ))[:n_parts]
            start_span =  (0,ticks[1])
            course_span = (ticks[1],ticks[-1])
            end_span = (ticks[-1],no_sequences)
            stage_ft = [np.vstack([ret_part_ft(1,start_span,ft),ret_part_ft(course[0],course_span,ft),ret_part_ft(course[1],course_span,ft),ret_part_ft(1,end_span,ft)])]
            stage_ft = np.array(stage_ft)
            res = model.predict(stage_ft)[0]
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

            if (res[pos] > threshold): 
                if len(sentence) > 0: 
                    if (actions[np.argmax(res)] != sentence[-1])&(res[last_max]<end_threshhold):
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

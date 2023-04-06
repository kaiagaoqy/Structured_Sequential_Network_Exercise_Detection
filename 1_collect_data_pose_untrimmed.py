import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import random
from utils_pose import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


mp_holistic = mp.solutions.pose # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('action_detection','test_2/MP_Data_untrimmed_resize') 

# Actions that we try to detect
'''
收集数据时，指定3类动作
每类动作取10段视频clip的300张frame
'''
actions = np.array(['stride', 'squat', 'up', 'down'])

video_dic = os.listdir(os.path.join('/home/sstc/文档/action_detection/test_2/', '广播体操动作'))
'''
for video in video_dic:
    try:
        os.makedirs(os.path.join(DATA_PATH, video.replace('.avi','')))
    except:
        pass
'''

## 因视频不定长度，把是视频平均分成50份，每份取1帧

# Thirty videos worth of data
#no_sequences = 15

# Videos are going to be 30 frames in length
#sequence_length = 1


#cap = cv2.VideoCapture(0)

for  video in video_dic:
    cap = cv2.VideoCapture(os.path.join('/home/sstc/文档/action_detection/test_2/', '广播体操动作',video))
    
    frames_num_all=int(cap.get(7))   ##总帧数
        #avg_len =  int(frames_num_all/no_sequences) ## 每段长度
        #start_li = np.arange(0,frames_num_all-1,avg_len)[:sequence]## 每段视频的起始点

        # Set mediapipe model 
    with mp_holistic.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
            # NEW LOOP
            # Loop through actions
            #for action in actions:
            # Loop through sequences aka videos
            #cnt=0 ##clip计数
            #for start in range(no_sequences): ## 每段clip分帧数读取landmarks，再扩大感受野，压缩为(1664, )
                #start = np.random.randint(0,avg_len)
                #select_frames =  np.arange(start,frames_num_all-1,avg_len)[:no_sequences]
                #keypoint_union = np.zeros(132)
                
        for frame_num in range(frames_num_all):


            ret, frame = cap.read()
            frame = cv2.resize(frame,(540,960))
            image, results = mediapipe_detection(frame, holistic)
                #print(results)

                # Draw landmarks
            draw_styled_landmarks(image, results)
            curr = frame_num
                # NEW Apply wait logic

            cv2.putText(image, 'Collecting frames for Video Number {} || {}'.format(video.replace('.avi',''),curr,video), (15,12), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # Show to screen
            #cv2.namedWindow("OpenCV Feed'", 0) 
           # cv2.resizeWindow('OpenCV Feed',480,800)
            cv2.imshow('OpenCV Feed', image)

            keypoints = extract_keypoints(results)
            if frame_num==0:
                npy_keypoint = keypoints
            if frame_num>0:
                npy_keypoint = np.vstack([npy_keypoint,keypoints])
                if  frame_num == frames_num_all-1:
                    npy_path =os.path.join(DATA_PATH,video.replace('.mp4',''))
                    np.save(npy_path, npy_keypoint)
                    
                    #cv2.imwrite('/home/sstc/文档/action_detection/test/frames/{}/{}/{}.png'.format(action,video.replace('.avi',''),frame_num),image)
                    
                # Break gracefully
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break       
               # cnt+=1    
        cap.release()
cv2.destroyAllWindows()
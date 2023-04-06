
from matplotlib.contour import ContourSet
from utils_pose import *

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

mp_holistic = mp.solutions.pose # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('test_2/MP_Data_untrimmed_resize') 

# Actions that we try to detect
actions = np.array(['bg','stride', 'squat', 'up', 'down'])
CATE = 4
# Thirty videos worth of data
parts = 1
no_sequences = 25
stride =1
# Videos are going to be 30 frames in length
sequence_length = 1
def ret_part_ft(n_part,span,ft,parts=5):
    stage_ft = ft[span[0]:span[1],:]
    duration = stage_ft.shape[0]
    ticks = range(0,duration+1000,int(duration/n_part))
    for i in range (n_part):
        if i == 0:
            part_ft = ft[ticks[i]:ticks[i+1],:].mean(axis=0)/parts
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

class Classifiers(tf.keras.Model):
    def __init__(self,class_num_without_bg):
        super().__init__()
        ## Complete Classifier
        self.class_num_without_bg = class_num_without_bg
        #tf.layers.Input()
        self.lstm_cc = tf.keras.layers.LSTM(units=64,activation=tf.nn.relu,name='lstm_cc',input_shape=(None,5,132)) ##(128,256)  (64,256)  (256,)
        self.dense_cc = tf.keras.layers.Dense(units=self.class_num_without_bg,name='dense_cc') ## w=(100, 5) b=(5,)
        ## Activity Classifier
        self.lstm1_ac = tf.keras.layers.LSTM(units=64,activation=tf.nn.relu,return_sequences=True,input_shape=(None,3,132))##(132,256)  (64,256)  (256,)
        self.lstm2_ac = tf.keras.layers.LSTM(units=128,activation=tf.nn.relu,return_sequences=True)##(128,512)  (64,512)  (512,)
        self.lstm3_ac = tf.keras.layers.LSTM(units=64,activation=tf.nn.relu) ##(128,256)  (64,256)  (256,)
        self.dens1_ac =  tf.keras.layers.Dense(units=100,activation=tf.nn.relu) ## w = (64,100) b=(100,)
        self.dense2_ac = tf.keras.layers.Dense(units=self.class_num_without_bg+1) ## w=(100, 5) b=(5,)

    @tf.function
    def call(self, input_ac,input_cc):
        x_ac=self.lstm1_ac(input_ac) ## TensorShape([50, 5, 64])
        x_ac=self.lstm2_ac(x_ac) ## TensorShape([50, 5, 128])
        x_ac=self.lstm3_ac(x_ac) ## TensorShape([50, 64])
        x_ac=self.dens1_ac(x_ac) ## TensorShape([50, 100])
        x_ac=self.dense2_ac(x_ac) ## TensorShape([50, 5])
        out_ac = tf.nn.softmax(x_ac) ## TensorShape([50, 5])

        x_cc=self.lstm_cc(input_cc) ## TensorShape([50, 100])
        x_cc=self.dense_cc(x_cc) ## TensorShape([50, 5])
        out_cc = tf.nn.softmax(x_cc) ## TensorShape([50, 5])
        return out_ac,out_cc

threshold = 0.7
end_threshhold = 0.1

model = Classifiers(CATE) ##导入模型
checkpoint = tf.train.Checkpoint(myModel=model)             # 实例化Checkpoint，指定恢复对象为model
checkpoint.restore(tf.train.latest_checkpoint('/home/sstc/文档/action_detection/test_2/'))    # 从文件恢复模型最新参数



#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('/home/sstc/文档/action_detection/test_2/广播体操动作/VID20220328180027.mp4')

# Set mediapipe model 
with mp_holistic.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()
        #frame = cv2.resize(frame,(540,960))

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        #print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)

        sequence.append(keypoints)
        sequence = sequence[-no_sequences:]
        if  int(len(sequence)) >= no_sequences:
            a = np.expand_dims(sequence, axis=0)
            ft = a[0]
            n_parts = 2+course[1]
            ticks = range(0,no_sequences,int(no_sequences/n_parts ))[:n_parts]
            start_span =  (0,ticks[1])
            course_span = (ticks[1],ticks[-1])
            end_span = (ticks[-1],no_sequences)
            stage_ft = [np.vstack([ret_part_ft(1,start_span,ft,parts=parts),ret_part_ft(course[0],course_span,ft,parts=parts),ret_part_ft(course[1],course_span,ft,parts=parts),ret_part_ft(1,end_span,ft,parts=parts)])]
            ft_cc = np.array(stage_ft) ## (1, 5, 132)
            ft_ac= np.array(stage_ft)[:,1:-1,:] 
            res_ac,res_cc = model.call(ft_ac,ft_cc)
            res_cc_val = np.hstack([[[np.max(res_cc.numpy())]],res_cc.numpy()])
            res = tf.add(0.7*res_ac,0.3*res_cc_val).numpy()[0]
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

from pickle import FALSE
import cv2
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

mp_holistic = mp.solutions.pose # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('action_detection','test_2','MP_Data') 
MODEL = 'pose'
OPTIMIZER = 'adam'
# Actions that we try to detect
#actions = np.array(['1', '2', '3'])
actions = np.array(['stride', 'squat', 'up', 'down'])
course=(1,2) ## 两层金字塔
video_dic = {}
for  action in actions:
    video_dic[action] = os.listdir(os.path.join('/home/sstc/文档/action_detection/test_2/clipps/', action))
# Thirty videos worth of data
no_sequences = 8
stride = 1

# Videos are going to be 30 frames in length
#sequence_length = 1

label_map = {label:num for num, label in enumerate(actions)}
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

sequences, labels = [], []
for action in actions:
    for video in  video_dic[action] :
        window = []
        f_li = os.listdir(os.path.join(DATA_PATH, action,video.replace('.avi','')))
        f_li = np.sort(pd.Series(f_li).str.split('.',expand=True)[0].astype('int32'))
        cnt = 0 
        for i in f_li:
            if cnt==0:
                ft = np.load(os.path.join(DATA_PATH, action,video.replace('.avi',''),'{}.npy'.format(i)))
            else:
                ft = np.vstack([ft,np.load(os.path.join(DATA_PATH, action,video.replace('.avi',''),'{}.npy'.format(i)))])
            cnt+=1
        '''
        frame_max = f_li.max()
        if frame_max<no_sequences:
            selected = np.sort(np.array(list(range(frame_max))*int(np.ceil(no_sequences/frame_max))))[list(range(0,no_sequences,stride))]
        else:
            selected = list(range(0,no_sequences,stride))
        for frame_num in selected:
            #res = np.load(os.path.join(DATA_PATH, action, video.replace('.avi',''), "{}.npy".format(frame_num)))
            res = np.load(os.path.join(DATA_PATH, action, video.replace('.avi',''), "{}.npy".format(frame_num)))
            window.append(res)
        '''
        duration = ft.shape[0]
        epoch = 2+np.array(course).sum()
        n_part = 2+course[1]
        stride  = int(duration/n_part)
        ticks = range(0,duration,stride)
        start_span = (0,ticks[1])
        course_span = (ticks[1],ticks[-1])
        end_span = (ticks[-1],duration)
        

        sequences.append(np.vstack([ret_part_ft(1,start_span,ft),ret_part_ft(course[0],course_span,ft),ret_part_ft(course[1],course_span,ft),ret_part_ft(1,end_span,ft)]))
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
shape = X.shape[1:]
#print('x, y:',X.shape,y.shape) #x, y: (99, 49, 132) (99, 3)
## x：99段video，每个video被拆成49个clip，每个clip132个特征
## y:   99段ideo，3类动作转换为one-hot

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=1234, shuffle=True,stratify=y)

log_dir = os.path.join('/home/sstc/文档/action_detection/test_2/','Logs_ssn'+MODEL)
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=shape))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))
#model.load_weights('/home/sstc/文档/action_detection/ActionDetection-20220313/action.h5')
#model.add(Dense(actions.shape[0], activation='softmax'))
optimizer = optimizers.Adam(lr=1e-3)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

history = model.fit(X_train, y_train, epochs=300,batch_size=int(y .shape[0]/5),validation_data=(X_test,y_test), callbacks=[tb_callback])

model.summary()

model.save('/home/sstc/文档/action_detection/test_2/action_{}_mean_{}_50.h5'.format(MODEL,OPTIMIZER))

model.load_weights('/home/sstc/文档/action_detection/test_2/action_{}_mean_{}_50.h5'.format(MODEL,OPTIMIZER))

yhat = model.predict(X_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

mc = multilabel_confusion_matrix(ytrue, yhat)

acc = accuracy_score(ytrue, yhat)

print('multilabel_confusion_matrix:',multilabel_confusion_matrix)
print('acc:',acc)










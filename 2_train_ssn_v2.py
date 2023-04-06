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
from pose_embedding import  FullBodyPoseEmbedder
mp_holistic = mp.solutions.pose # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('action_detection','test_2','MP_Data_untrimmed') 
MODEL = 'pose'
OPTIMIZER = 'adam'
# Actions that we try to detect
#actions = np.array(['1', '2', '3'])
actions = np.array(['stride', 'squat', 'up', 'down'])


X_global = np.load('/home/sstc/文档/action_detection/test_2/out/global_ft.npy') ##(833, 5, 132)
X_course = np.load('/home/sstc/文档/action_detection/test_2/out/global_ft_course.npy') ## (833,)
labels = np.load('/home/sstc/文档/action_detection/test_2/out/global_y.npy') ##(833, 3, 132)

Y_global = to_categorical(labels).astype(int)
shape = X_global.shape[1:] ##(833, 5)  5类（4类动作+背景类）




import tensorflow as tf

X_global = tf.constant(X_global,name='X_global')
X_course = tf.constant(X_course,name='X_course')
Y_global= tf.constant(Y_global,name='Y_global')





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

#model.save('/home/sstc/文档/action_detection/test_2/action_{}_mean_{}_50.h5'.format(MODEL,OPTIMIZER))
#model.load_weights('/home/sstc/文档/action_detection/test_2/action_{}_mean_{}_50.h5'.format(MODEL,OPTIMIZER))
model.save('/home/sstc/文档/action_detection/test_2/action_{}_mean_{}_50.h5'.format(MODEL,OPTIMIZER))
model.load_weights('/home/sstc/文档/action_detection/test_2/action_{}_mean_{}_50.h5'.format(MODEL,OPTIMIZER))

yhat = model.predict(X_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

mc = multilabel_confusion_matrix(ytrue, yhat)

acc = accuracy_score(ytrue, yhat)

print('multilabel_confusion_matrix:',multilabel_confusion_matrix)
print('acc:',acc)










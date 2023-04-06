
from turtle import shape
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os

###超参数声明
DATA_PATH = '/home/sstc/文档/action_detection/test_2/Models/combine/5'


actions = np.array(['stride', 'squat', 'up', 'down'])
X_global = np.load('/home/sstc/文档/action_detection/test_2/out/cpp_combine/5/global_ft.npy') ##(785, 5, 132)
X_course = np.load('/home/sstc/文档/action_detection/test_2/out/cpp_combine/5/global_ft_course.npy') ##(833, 3, 132) 
Y_global = np.load('/home/sstc/文档/action_detection/test_2/out/cpp_combine/5/global_y.npy') ##(785, )  without bg class
Y_global_course = np.load('/home/sstc/文档/action_detection/test_2/out/cpp_combine/5/global_y_course.npy')  ## (833,)
class_without_bg = len(np.unique(Y_global))
FEATURE_DIM = X_global.shape[2]


Y_global = to_categorical(Y_global,num_classes=class_without_bg+1,dtype='int64')[:,1:] ##(785, 4)  
Y_global_course = to_categorical(Y_global_course,num_classes=class_without_bg+1,dtype='int64')## (833,5)
#shape = X_global.shape[1:] ##(833, 5)  5类（4类动作+背景类）

X_global = tf.constant(X_global,name='X_global')
X_course = tf.constant(X_course,name='X_course')
Y_global= tf.constant(Y_global,name='Y_global')## (785,4)
Y_global_course= tf.constant(Y_global_course,name='Y_global_course')  ## (833,5)
#tf_is_training = tf.placeholder(tf.bool, None)  # to control dropout when training and testing

##  Classifier
num_epochs = 700
batch_size = 150
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

class Classifiers(tf.keras.Model):
    def __init__(self,class_num_without_bg):
        super().__init__()
        ## Complete Classifier
        self.class_num_without_bg = class_num_without_bg
        #tf.layers.Input()
        self.dense0_cc =  tf.keras.layers.Dense(units=300,activation=tf.nn.relu)
        self.bn_cc = tf.keras.layers.BatchNormalization()
        #self.bn_cc = tf.nn.dropout(self.bn_cc , rate=0.9)   # drop out 10% of inputs
        self.lstm_cc = tf.keras.layers.LSTM(units=64,activation=tf.nn.relu,name='lstm_cc',return_sequences=True,input_shape=(None,5,FEATURE_DIM)) ##(128,256)  (64,256)  (256,)
        self.lstm2_cc = tf.keras.layers.LSTM(units=128,activation=tf.nn.relu,return_sequences=True)##(128,512)  (64,512)  (512,)
        self.lstm3_cc = tf.keras.layers.LSTM(units=64,activation=tf.nn.relu) ##(128,256)  (64,256)  (256,)
        self.dense1_cc =  tf.keras.layers.Dense(units=100,activation=tf.nn.relu) ## w = (64,100) b=(100,)
        self.dense2_cc = tf.keras.layers.Dense(units=self.class_num_without_bg,name='dense_cc') ## w=(100, 5) b=(5,)
        ## Activity Classifier
        self.dense0_ac =  tf.keras.layers.Dense(units=300,activation=tf.nn.relu)
        self.bn_ac = tf.keras.layers.BatchNormalization()
        #self.bn_ac = tf.nn.dropout(self.bn_ac , rate=0.9)   # drop out 10% of inputs
        self.lstm1_ac = tf.keras.layers.LSTM(units=64,activation=tf.nn.relu,return_sequences=True,input_shape=(None,3,FEATURE_DIM))##(132,256)  (64,256)  (256,)
        self.lstm2_ac = tf.keras.layers.LSTM(units=128,activation=tf.nn.relu,return_sequences=True)##(128,512)  (64,512)  (512,)
        self.lstm3_ac = tf.keras.layers.LSTM(units=64,activation=tf.nn.relu) ##(128,256)  (64,256)  (256,)
        self.dens1_ac =  tf.keras.layers.Dense(units=100,activation=tf.nn.relu) ## w = (64,100) b=(100,)
        self.dense2_ac = tf.keras.layers.Dense(units=self.class_num_without_bg+1) ## w=(100, 5) b=(5,)

    @tf.function
    def call(self, input_ac,input_cc,training=False):
        x_ac =   self.dense0_ac(input_ac)
        x_ac = tf.nn.dropout(x_ac , rate=0.2)   # drop out 10% of inputs
        x_ac = self.bn_ac(x_ac,training=training)
        x_ac=self.lstm1_ac(x_ac) ## TensorShape([50, 5, 64])
        x_ac=self.lstm2_ac(x_ac) ## TensorShape([50, 5, 128])
        x_ac=self.lstm3_ac(x_ac) ## TensorShape([50, 64])
        x_ac = tf.nn.dropout(x_ac , rate=0.2)   # drop out 10% of inputs
        x_ac=self.dens1_ac(x_ac) ## TensorShape([50, 100])
        x_ac=self.dense2_ac(x_ac) ## TensorShape([50, 5])
        out_ac = tf.nn.softmax(x_ac) ## TensorShape([50, 5])

        x_cc=self.dense0_ac(input_cc)
        x_cc = tf.nn.dropout(x_cc , rate=0.2)   # drop out 10% of inputs
        x_cc = self.bn_cc(x_cc,training=training)
        x_cc=self.lstm_cc(x_cc) ## TensorShape([50, 100])
        x_cc=self.lstm2_cc(x_cc) 
        x_cc=self.lstm3_cc(x_cc) 
        x_cc = tf.nn.dropout(x_cc , rate=0.2)   # drop out 10% of inputs
        x_cc=self.dense1_cc(x_cc) 
        x_cc=self.dense2_cc(x_cc) ## TensorShape([50, 5])
        out_cc = tf.nn.softmax(x_cc) ## TensorShape([50, 5])
        return out_ac,out_cc
'''
class ActivityClassifier(tf.keras.Model):
    def __init__(self,class_num):
        super().__init__()
        self.class_num = class_num
        ## return_sequences:默认为false。当为false时，返回最后一层最后一个步长的hidden state;当为true时，返回最后一层的所有hidden state。
        ##r eturn_state:默认false.当为true时，返回最后一层的最后一个步长的输出hidden state和输入cell state。
        self.lstm1 = tf.keras.layers.LSTM(units=64,activation=tf.nn.relu,return_sequences=True)##(132,256)  (64,256)  (256,)
        self.lstm2 = tf.keras.layers.LSTM(units=128,activation=tf.nn.relu,return_sequences=True)##(128,512)  (64,512)  (512,)
        self.lstm3 = tf.keras.layers.LSTM(units=64,activation=tf.nn.relu) ##(128,256)  (64,256)  (256,)
        self.dens1 =  tf.keras.layers.Dense(units=100,activation=tf.nn.relu) ## w = (64,100) b=(100,)
        self.dense2 = tf.keras.layers.Dense(units=class_num) ## w=(100, 5) b=(5,)

    ## call函数需保存
    #@tf.function
    def call(self, input):
        x=self.lstm1(input) ## TensorShape([50, 5, 64])
        x=self.lstm2(x) ## TensorShape([50, 5, 128])
        x=self.lstm3(x) ## TensorShape([50, 64])
        x=self.dens1(x) ## TensorShape([50, 100])
        x=self.dense2(x) ## TensorShape([50, 5])
        out = tf.nn.softmax(x) ## TensorShape([50, 5])
        return out

class CompleteClassifier(tf.keras.Model):
    def __init__(self,class_num):
        super().__init__()
        self.class_num = class_num
        self.lstm = tf.keras.layers.LSTM(units=64,activation=tf.nn.relu) ##(128,256)  (64,256)  (256,)
        self.dense = tf.keras.layers.Dense(units=class_num) ## w=(100, 5) b=(5,)

    ## call函数需保存
    #@tf.function
    def call(self, input):
        x=self.lstm(input) ## TensorShape([50, 100])
        x=self.dense(x) ## TensorShape([50, 5])
        out = tf.nn.softmax(x) ## TensorShape([50, 5])
        return out
'''

def get_batch(train_data,train_label,train_course_data,train_course_label, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        num_train_data = train_data.shape[0]
        num_train_course_data = train_course_data.shape[0]
        index_cc = list(np.random.randint(0, num_train_data, batch_size))
        index_ac = list(np.random.randint(0, num_train_course_data, batch_size))
        return tf.constant(train_data.numpy()[index_cc]), tf.constant(train_label.numpy()[index_cc]),\
        tf.constant(train_course_data.numpy()[index_ac]), tf.constant(train_course_label.numpy()[index_ac])


#ac = ActivityClassifier(Y_global_course.shape[1]) ## 5
#cc = CompleteClassifier(Y_global.shape[1]) ## 4

classifier = Classifiers(Y_global.shape[1]) ##4
checkpoint = tf.train.Checkpoint(myModel=classifier,optimizer=optimizer)
## tensorbard地址
tensorboard_trnaddr = '/home/sstc/文档/action_detection/test_2/tensorbard/all'
tensorboard_acaddr = '/home/sstc/文档/action_detection/test_2/tensorbard/ac'
tensorboard_ccaddr = '/home/sstc/文档/action_detection/test_2/tensorbard/cc'
summary_writer_all = tf.summary.create_file_writer(tensorboard_trnaddr)     # 参数为记录文件所保存的目录
summary_writer_ac= tf.summary.create_file_writer(tensorboard_acaddr) 
summary_writer_cc = tf.summary.create_file_writer(tensorboard_ccaddr) 
#if not os.path.exists(tensorboard_trnaddr):
 #   os.makedirs(tensorboard_trnaddr)

num_batches = int(X_global.shape[0]//batch_size)
for epoch_idx in range(num_epochs):
    if  epoch_idx%10 ==0:
        print("-"*10,"epoch: %d"%(epoch_idx),'-'*10)
        for batch_idx in range (num_batches):
            '''
            x_cc:(batch_size,1,5,132)
            y_cc:(batch_size,5)
            x_ac:(batch_size,3,132)
            y_ac:(batch_size,4)
            '''
            x_cc,y_cc,x_ac,y_ac = get_batch(X_global,Y_global,X_course,Y_global_course,batch_size)
            with tf.GradientTape() as tape: ## grads len=13
                ## TensorShape([50, 5])
                y_pred_ac,y_pred_cc = classifier.call(x_ac,x_cc)
                loss_ac = tf.keras.losses.categorical_crossentropy(y_true=y_ac, y_pred=y_pred_ac) ## one-hot用categorical_crossentropy，数字编码用sparse_categorical_crossentropy
                loss_ac = tf.reduce_mean(loss_ac, name='loss_ac')
                loss_cc = tf.keras.losses.categorical_crossentropy(y_true=y_cc, y_pred=y_pred_cc) ## one-hot用categorical_crossentropy，数字编码用sparse_categorical_crossentropy
                loss_cc = tf.reduce_mean(loss_cc,name='loss_cc')
                loss = tf.add(loss_ac,0.3*loss_cc,name='loss_without_regression')
                # loss = act_loss + comp_loss * args.comp_loss_weight + reg_loss * args.reg_loss_weight
                
                print("batch %d:\n loss %f, loss_ac %f, loss_cc %f" % (batch_idx, loss.numpy(),loss_ac.numpy(),loss_cc.numpy()))

                sparse_categorical_accuracy_cc = tf.keras.metrics.CategoricalAccuracy()
                sparse_categorical_accuracy_ac = tf.keras.metrics.CategoricalAccuracy() 

                sparse_categorical_accuracy_cc.update_state(y_true=y_cc, y_pred=y_pred_cc)
                sparse_categorical_accuracy_ac.update_state(y_true=y_ac, y_pred=y_pred_ac)
                #print(sparse_categorical_accuracy_ac.result())
                sparse_categorical_accuracy = tf.reduce_mean([sparse_categorical_accuracy_ac.result(),sparse_categorical_accuracy_cc.result()])
                print("test accuracy -- avg : %f \n acc_ activity : %f   |--|  acc_complete : %f" % (sparse_categorical_accuracy.numpy(),sparse_categorical_accuracy_ac.result(),sparse_categorical_accuracy_cc.result()))
                #print('-'*25)
                        
            grads_ac = tape.gradient(loss, classifier.variables) # 13+5
            optimizer.apply_gradients(grads_and_vars=zip(grads_ac, classifier.variables)) ## ac.variables 13个 lstm各3个：kernal,recurrent_kernal,bias

    with summary_writer_all.as_default(): ##记录器
        tf.summary.scalar("Loss",loss.numpy(),step=epoch_idx,description='loss_all')
        tf.summary.scalar("Accuracy",sparse_categorical_accuracy.numpy(),step=epoch_idx,description='categorical_accuracy_avg')    
            
    with summary_writer_ac.as_default(): ##记录器
        tf.summary.scalar("Loss",loss_ac.numpy(),step=epoch_idx,description='loss_ac')
        tf.summary.scalar("Accuracy",sparse_categorical_accuracy_cc.result(),step=epoch_idx,description='categorical_accuracy_cc')
    with summary_writer_cc.as_default(): ##记录器
        tf.summary.scalar("Loss",loss_cc.numpy(),step=epoch_idx,description='loss_cc')
        tf.summary.scalar("Accuracy",sparse_categorical_accuracy_ac.result(),step=epoch_idx,description='categorical_accuracy_ac')

    '''
    kernel: (input_dim, unit * 4)   LSTM有IFCO（input, forget, candidate, ouput）四个门
    recurrent_kernel: (unit, unit * 4)  激活函数
    bias: (unit * 4)
    '''


## 训练完直接保存
tf.saved_model.save(classifier, os.path.join(DATA_PATH,'ssn_model_without_backbone'))
checkpoint.save(os.path.join(DATA_PATH,'ssn_model_without_backbone.ckpt'))
    





'''
##----------模型评估
sparse_categorical_accuracy_cc = tf.keras.metrics.SparseCategoricalAccuracy()
sparse_categorical_accuracy_ac = tf.keras.metrics.SparseCategoricalAccuracy()
num_batches = int(X_global.shape[0]// batch_size)
for batch_index in range(num_batches):
    start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
    y_pred_ac,y_pred_cc = classifier.predict(X_course[start_index: end_index],X_global[start_index: end_index])
    sparse_categorical_accuracy_cc.update_state(y_true=Y_global[start_index: end_index], y_pred=y_pred_cc)
    sparse_categorical_accuracy_ac.update_state(y_true=Y_global_course[start_index: end_index], y_pred=y_pred_ac)
    sparse_categorical_accuracy = tf.reduce_mean(sparse_categorical_accuracy_cc,sparse_categorical_accuracy_ac)
print("test accuracy -- avg : %f \n test accuracy -- activity : %f,    test accuracy -- complete : %f" % (sparse_categorical_accuracy,sparse_categorical_accuracy_ac.result(),sparse_categorical_accuracy_cc.result()))
'''
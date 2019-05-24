import numpy as np
import tensorflow as tf
from numpy import argmax
from sklearn.model_selection import train_test_split
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import scipy.io
from sklearn.model_selection import LeaveOneOut,KFold

weight_track = 0


   
def setupGPU():
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config )
    return sess

def input_fn(features, labels,batch_size,cast_type,shuffle_repeat = True):
  
    train_dataset = tf.data.Dataset.from_tensor_slices((features,labels))
    #files = tf.data.Dataset.list_files("")
    #dataset = tf.data.TFRecordDataset(files, num_parallel_reads = 32)
    if(shuffle_repeat):
        train_dataset =train_dataset.apply(tf.contrib.data.shuffle_and_repeat(10000 ))
    else:
        train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.apply(tf.contrib.data.prefetch_to_device("/gpu:0"))
    iterator = train_dataset.make_initializable_iterator()
    x,y = iterator.get_next()
    x = tf.cast(x,cast_type)

    return x,y,iterator

def define_accuracy_multi(logits,true_y):
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(true_y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc_summary = tf.summary.scalar("accuracy", accuracy)
    return accuracy,acc_summary

def define_train_step(optimizer,learning_rate,loss):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        with tf.name_scope("train"):
            train_step = optimizer(learning_rate).minimize(loss)
    return train_step


def fullyCon( x,num_nodes,batch_norm = False, activation = None,is_train=True,name = 'fc',bn_name='batch_normalization',l2 = False,l2_name='l2'):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([x.get_shape().as_list()[1],num_nodes], stddev=0.1),name = 'W')
        b = tf.Variable(tf.constant(0.1, shape=[num_nodes]),name='B') 
        fcl= tf.matmul(x, w) + b
        if(batch_norm):
            fcl = tf.layers.batch_normalization(fcl,training=is_train, name=bn_name)
        if( activation == 'relu'):
            fcl = tf.nn.relu(fcl)
        elif (activation == 'softmax' and batch_norm == False):
            fcl = tf.nn.softmax(fcl)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", fcl)
        print(fcl)
        if l2:
            return fcl,tf.nn.l2_loss(w,name=l2_name)
        else:
            return fcl,0


def fc_layers(x,netSpec,bn = False,use_dropout = False,keep_prob= 0.8,is_train=True,use_l2=False):
    l2_sum = 0

    fc,l2_1 = fullyCon(x,netSpec[0],bn,'relu',is_train = is_train,name = 'fc1',bn_name='bn1',l2=use_l2,l2_name='l2_1')
    l2_sum += l2_1
   
    for i in range(len(netSpec)-1):
        
        fc,l2_term = fullyCon(fc,netSpec[i+1],bn,'relu',is_train = is_train,name='fc'+ str(i+2),bn_name = 'bn'+str(i+2),l2 = use_l2,l2_name='l2_'+str(i+2))
        l2_sum += l2_term
        
        if(use_dropout and is_train is not None):
            fc = tf.cond(is_train, lambda: tf.nn.dropout(fc,keep_prob,name='Dropout'+str(i+2)),lambda:tf.nn.dropout(fc,1.0,name = 'Dropout'+str(i+2)))

    return fc,l2_sum

def output_layer(x,n_classes, name= 'output_layer'):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([ x.get_shape().as_list()[1],n_classes], stddev=0.1),name= 'W' )
        b = tf.Variable(tf.constant(0.1, shape=[n_classes]),name = 'B') 
        act= tf.matmul(x, w) + b
        y_NN= tf.nn.softmax(act)
        y_NN
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", y_NN)
        return y_NN

def loss_func_multi(logits,true_y,l2_loss = 0,l2_factor=0):
    with tf.name_scope("xent"):
        xent = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2( logits=logits, labels=true_y), name="xent")
        tf.summary.scalar("xent", xent)
        loss = tf.reduce_mean(xent + l2_loss*l2_factor)
        tf.summary.scalar('loss_w_l2',loss)
        return loss
def loss_func_binary(logits,true_y):
    with tf.name_scope("xent"):
        xent = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( logits=logits, labels=true_y), name="xent")
        tf.summary.scalar("xent", xent)
        return xent

def conv_layer2D(x_image,size_in, size_out,ksize =5,Dformat = "NCHW", batch_norm = False,is_train=True, name="conv2D",pname = 'Max Pool' ,bn_name = 'batch_normalization_conv'):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([ksize, ksize, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(x_image, w, strides=[1, 1, 1, 1], padding="SAME",data_format=Dformat)
        res = tf.nn.bias_add(conv,b,data_format=Dformat)
        if(batch_norm ):
            res = tf.layers.batch_normalization(res,training=is_train, name=bn_name)
        act = tf.nn.relu(res)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        if(Dformat == "NHWC"):
            return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME",data_format= Dformat)
        else:
            return tf.nn.max_pool(act, ksize=[1, 1, 2, 2], strides=[1, 1, 2, 2], padding="SAME",data_format= Dformat)

def conv_layer1D(x,fWidth, size_in,size_out,Dformat ='NCW',batch_norm = False,name = 'conv1D',pname = 'Max_Pool1D'):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([fWidth,size_in,size_out],stddev=0.1),name = 'w')
        b = tf.Variable(tf.constant(0.1,shape=[size_out]),name = 'B')

        conv = tf.nn.conv1d(x,w,stride =1,padding = 'SAME',data_format = Dformat)
        res = tf.nn.bias_add(conv,b,data_format='NCHW')
        act = tf.nn.relu(res )
        tf.summary.histogram("weights", w)
        #tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        if(Dformat == 'NCW'):
            df = 'channels_first'
        else:
            df = 'channels_last'
        return tf.layers.max_pooling1d(act, pool_size=[2], strides=[2], padding="SAME",data_format= df,name = pname)

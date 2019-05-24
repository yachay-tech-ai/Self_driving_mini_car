import cv2
import numpy as np
import tensorflow as tf
import glob
import sys
import time
import os
from sklearn.model_selection import train_test_split
import Tensorflow_neural_methods_procedural as tfnm

LOGDIR = '/SELF/'

def load_data(input_size, path):
    print("Loading training data...")
    start = time.time()

    # load training data
    X = np.empty((0, input_size))
    y = np.empty((0, 4))
    training_data = glob.glob(path)

    # if no data, exit
    if not training_data:
        print("Data not found, exit")
        sys.exit()

    for single_npz in training_data:
        with np.load(single_npz) as data:
            train = data['train']
            train_labels = data['train_labels']
        X = np.vstack((X, train))
        y = np.vstack((y, train_labels))

    print("Image array shape: ", X.shape)
    print("Label array shape: ", y.shape)

    end = time.time()
    print("Loading data duration: %.2fs" % (end - start))

    # normalize data
    #X = X / 255.

    # train validation split, 7:3
    return train_test_split(X, y, test_size=0.3)

class ConvolutionalNeuralNetwork(object):

    def __init__(self,input_size,shuffle_repeat):
        self.sess = tfnm.setupGPU()
        self.input_size = input_size
        self.X = tf.placeholder(tf.float32, shape = [None,input_size])
        self.Y = tf.placeholder(tf.float32, shape = [None,4])
        self.is_train = tf.placeholder(tf.bool)
        self.batch_size = tf.placeholder(tf.int64)
        self.x,self.y,self.iterator = tfnm.input_fn(self.X,self.Y,self.batch_size,tf.float32,shuffle_repeat=shuffle_repeat)
    
    def create_graph(self,netspec,conv_layers,learning_rate,optimizer = tf.train.AdamOptimizer,ksize =5):
        global LOGDIR
        copl= 1
        size_in = 1
        x_image = tf.reshape(self.x,[-1,320,120,1])
        self.conv = tfnm.conv_layer2D(x_image,ksize = ksize,size_in=size_in,size_out=3,Dformat = 'NHWC',batch_norm = True,name = "Conv2D_"+ str(1),pname = 'Max Pool_'+str(1) ,bn_name = 'batch_normalization_conv'+str(1))
        for i in range(conv_layers-1):
            size_in = 3*(i+1)
            copl = 3*(i+2)
            self.conv = tfnm.conv_layer2D(self.conv,size_in,copl,Dformat = 'NHWC',batch_norm = True,name = "Conv2D_"+str(i+1),pname = 'Max Pool_'+str(i+1) ,bn_name = 'batch_normalization_conv'+str(i+2))
        self.flatten = tf.reshape(self.conv,[-1,int(self.input_size*copl/(2**(2*conv_layers)))])
        self.fc,l2_ = tfnm.fc_layers(self.flatten,netspec,bn = True,use_dropout = True,keep_prob = 0.8,is_train = self.is_train)
        self.logits,l2_ = tfnm.fullyCon(self.fc,4,batch_norm=True,is_train=self.is_train)
        self.scaled = tf.nn.softmax(self.logits,name='softmax')
        self.loss = tfnm.loss_func_multi(self.logits,self.y)

        self.accuracy,self.acc_summary = tfnm.define_accuracy_multi(self.scaled,self.y)
        self.train_step = tfnm.define_train_step(optimizer,learning_rate,self.loss)

        self.summ = tf.summary.merge_all()
        self.sess.run(tf.global_variables_initializer())
        self.trainwriter = tf.summary.FileWriter(LOGDIR+'train')
        self.trainwriter.add_graph(self.sess.graph)
        self.testwriter = tf.summary.FileWriter(LOGDIR+'test')
        self.testwriter.add_graph(self.sess.graph)

    def train(self,data_train,labels_train,epochs):
        start = time.time()
        self.sess.run(self.iterator.initializer,feed_dict={self.X:data_train,self.Y:labels_train,self.batch_size:100,self.is_train:True})
        
        for i in range(int(epochs*data_train.shape[0]/self.batch_size.eval(feed_dict = {self.batch_size:30}))):
            if i%100 == 0:
                [self.train_accuracy, s] = self.sess.run([self.accuracy, self.summ],feed_dict={self.is_train:True})
                self.trainwriter.add_summary(s,i)
                print("step %d, training accuracy %g"%(i, float(self.train_accuracy)))
            self.train_step.run(feed_dict={self.is_train:True})
        end = time.time()
        print("Training duration: %.2fs" % (end - start))
    
    def evaluate(self,data_test,labels_test):
        self.sess.run(self.iterator.initializer,feed_dict={self.X:data_test,self.Y:labels_test,self.batch_size:labels_test.shape[0],self.is_train:False})
        summary,val_acc = self.sess.run([self.acc_summary,self.accuracy],feed_dict={self.is_train:False})
        self.testwriter.add_summary(summary,0)
        print("Validation accuracy %g"%val_acc) 

    def predict(self,data_point,label):
        start = time.time()
        self.sess.run(self.iterator.initializer,feed_dict={self.X:data_point,self.Y:label,self.batch_size:1,self.is_train:False})
        pred = self.scaled.eval(feed_dict={self.is_train:False})
        pred = np.argmax(pred,axis=-1)
        end = time.time()
        print('Prediction duration: %.2fs' % (end - start))
        return pred

    def reset(self):
        self.sess.run(tf.global_variables_initializer())
    
    def close(self):
        self.sess.close()


from tf_model import load_data, ConvolutionalNeuralNetwork
import tensorflow as tf
import numpy as np
import time

input_size = 120*320
data_path = "training_data/*.npz"
files = ["training_data/1557686332.npz","training_data/1557686416.npz","training_data/1557686463.npz","training_data/1557686536.npz","training_data/1557686605.npz","training_data/1557686773.npz"]
times_t = []
times_v = []
steps = [32,50,21,10,42,61]



#neural network trainning
netspec = [1164,100,50,10]

neural_net = ConvolutionalNeuralNetwork(input_size,True)
neural_net.create_graph(netspec,3,0.01)

for f in files:
    X_train, X_valid, y_train, y_valid = load_data(input_size, f)
    t1 = time.time()
    neural_net.train(X_train,y_train,100)
    t = time.time() - t1
    times_t.append(t)
    t1 = time.time()
    neural_net.evaluate(X_valid,y_valid)
    t = time.time() - t1
    times_v.append(t)
    neural_net.reset()

for i in range(len(steps)):
    print("{} : for train {} , for val {}".format(steps[i],times_t[i],times_v[i]))


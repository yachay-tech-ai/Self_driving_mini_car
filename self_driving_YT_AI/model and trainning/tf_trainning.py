from tf_model import load_data, ConvolutionalNeuralNetwork
import tensorflow as tf
import numpy as np

input_size = 120*320
data_path = "training_data/*.npz"

#dataset lading
X_train, X_valid, y_train, y_valid = load_data(input_size, data_path)

#neural network trainning
netspec = [200,200]

neural_net = ConvolutionalNeuralNetwork(input_size,True)
neural_net.create_graph(netspec,3,0.01)
neural_net.train(X_train,y_train,100)

neural_net.evaluate(X_valid,y_valid)





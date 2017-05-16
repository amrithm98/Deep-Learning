from __future__ import print_function
import tensorflow as tf

#Get data online and save it
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('/tmp/data',one_hot=True)	

#hyper parameters
learning_rate=0.0001
training_iters=200000
batch_size=128
display_step=50

#network Parameters
n_input=784 #28X28
n_classes=10 #10 Digits
dropout=0.75


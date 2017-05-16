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
dropout=0.75	#deactivating neurons randomly -> prevents overfitting => Enabling it to generalize

#Placeholders
x=tf.placeholder(tf.float32,[None,n_input])
y=tf.placeholder(tf.float32,[None,n_classes])
keep_prob=tf.placeholder(tf.float32)	#for dropout

#convolution Layer==>Increasing Abstraction
def conv2d(x,W,b,strides=1):
	x=tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
	x=tf.nn.bias_add(x,b)
	return tf.nn.relu(x)

#Max Pooling Layer ==>Single Output from a pool taking maximum of the neurons
def maxPool2d(x,k=2):
	return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1])




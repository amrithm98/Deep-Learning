from __future__ import print_function
import tensorflow as tf

#Get data online and save it
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('/tmp/data',one_hot=True)	

#hyper parameters
learning_rate=0.0001
training_iters=200000
batch_size=128
display_step=10

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
	return tf.nn.relu(x)	#Rectified Linear Units - Activation

#Max Pooling Layer ==>Single Output from a pool taking maximum of the neurons
def maxPool2d(x,k=2):
	return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')


#Model
#--------
#convoluton Layer 1 ==> Max Pooling 1 ==> Convolution Layer 2 ==> Max Pooling 2 ==>Fully connected Layer1 ==>Output

#Create a Model
def convNet(x,weights,biases,dropout):
	
	#reshaping input data
	x=tf.reshape(x,shape=[-1,28,28,1])
	
	#convolutional Layer1
	conv1=conv2d(x,weights['wc1'],biases['bc1'])
	
	#Apply Max Pooling to conv1==>DownSampling
	conv1=maxPool2d(conv1,k=2)
	
	#convolutional Layer 2 ==> Input is convolutional Layer 1
	conv2=conv2d(conv1,weights['wc2'],biases['bc2'])

	#Apply Max Pooling to conv2==>DownSampling
	conv2=maxPool2d(conv2,k=2)

	#Dense Layer ==> Fully connected Layers ==> Every neuron connected to every neuron in previous layers where previous layers are 
	#convolutional layers
	fc1=tf.reshape(conv2,[-1,weights['wd1'].get_shape().as_list()[0]])
	fc1=tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
	fc1=tf.nn.relu(fc1)

	#Apply dropout
	fc1=tf.nn.dropout(fc1,dropout)
	
	#Output Class Predcition
	out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
	return out

#Weights
with tf.device("/cpu:0"):
	weights={'wc1':tf.Variable(tf.random_normal([5,5,1,32])),
		 'wc2':tf.Variable(tf.random_normal([5,5,32,64])),
		 'wd1':tf.Variable(tf.random_normal([7*7*64,1024])),
		 'out':tf.Variable(tf.random_normal([1024,n_classes]))}

#biases
with tf.device("/cpu:0"):
	biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
saver = tf.train.Saver()
#Construct Model
prediction=convNet(x,weights,biases,keep_prob)

#Optimizer and loss functions
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
#AdamOptimizer,Adagrad etc are good optimizers
Optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#evaluate model
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))

#accuracy
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#Initialize variables
init=tf.global_variables_initializer()

#launch the graph

with tf.Session() as sess:
	sess.run(init)
	step=1
	#print ('Inital Weights: ','WC1',sess.run(weights['wc1']),'WC2',sess.run(weights['wc2']),'WD1',sess.run(weights['wd1']),'Out',sess.run(weights['out']))
	#train until max iters
	while step*batch_size <training_iters:
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		#Optimisation/Back Propogation
		sess.run(Optimizer,feed_dict={x:batch_x,y:batch_y,keep_prob:dropout})
		if step%display_step==0:
			loss,acc=sess.run([cost,accuracy],feed_dict={x:batch_x,y:batch_y,keep_prob:1.})
			print("Iter "+str(step*batch_size)+",Minibatch Loss="+ \
			"{:.6f}".format(loss)+ ", Training Accuracy= " + \
			"{:.5f}".format(acc))	
		step+=1
	#print ('Final Weights: ','WC1',sess.run(weights['wc1']),'WC2',sess.run(weights['wc2']),'WD1',sess.run(weights['wd1']),'Out',sess.run(weights['out']))
	save_path = saver.save(sess, "/tmp/model.ckpt")
	print("Model saved in file: %s" % save_path)	
	print("Optimisation Done")	
	print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],keep_prob: 1.}))

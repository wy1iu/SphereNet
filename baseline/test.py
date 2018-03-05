import numpy as numpy
import tensorflow as tf
from loss import loss2
from cifar10_input import *
from vgg_bn import VGG

n_class = 10
batch_sz = 125
max_epoch = 10000/125
data_path = '/home/yanming/dataset/cifar10/'
model_file = './models/cifar10_'

is_training = tf.placeholder("bool")
images, labels = inputs(True, data_path, batch_sz)
vgg = VGG()
vgg.build(images, n_class, is_training)
acc_op = tf.reduce_mean(tf.to_float(tf.equal(labels, tf.to_int32(vgg.pred))))


sess = tf.Session()
tf.train.Saver().restore(sess, model_file)
print("------restore from: %s" % model_file)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

acc = 0
for i in xrange(max_epoch):
	acc = acc + sess.run(acc_op, {is_training: False})

print('accuracy=%.4f' % (acc/max_epoch))
		





from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def loss2(logits, labels, num_classes, scope, head=None):
    with tf.name_scope(scope):
        logits = tf.reshape(logits, (-1, num_classes))
        softmax = tf.nn.softmax(logits) + 1e-4

        labels = tf.to_float(tf.one_hot(tf.reshape(labels, [-1]), num_classes))
        eps = 1e-2
        labels = (1-eps)*tf.to_float(tf.reshape(labels, (-1, num_classes))) + eps/num_classes

        if head is not None:
            cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax),
                                           head), reduction_indices=[1])
        else:
            cross_entropy = -tf.reduce_sum(
                labels * tf.log(softmax), reduction_indices=[1])

    return tf.reduce_mean(cross_entropy)    

import numpy as np
import tensorflow as tf

a = tf.constant([0,1,0,0,1,1])
b = tf.constant([[0,1],[1,1],[0,0],[1,0],[0,0],[1,1]])

a = tf.cast(a, tf.int32)
b = tf.cast(b, tf.float32)

print (tf.nn.sparse_softmax_cross_entropy_with_logits(labels=a, logits=b))

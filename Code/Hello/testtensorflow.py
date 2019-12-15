# 1 import 2 string 3 print
import tensorflow as tf

g = tf.Graph()
with g.as_default():
    hello = tf.constant('hello tf1')
    sess = tf.compat.v1.Session(graph=g)
    print(sess.run(hello))

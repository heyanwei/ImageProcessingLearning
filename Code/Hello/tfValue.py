# 常用数据类型的说明
import tensorflow as tf

g = tf.Graph()
with g.as_default():
    sess = tf.compat.v1.Session(graph=g)
    init_op = tf.compat.v1.global_variables_initializer()

    data1 = tf.constant(2.5) # 常量
    data1_1 = tf.constant(2, dtype=tf.int32)

    data2 = tf.Variable(10,name='var') # 变量
    # 变量需要初始化   
    sess.run(init_op)
    
    print(sess.run(data1))
    print(sess.run(data1_1))
    # print(sess.run(data2))
# 常用数据类型的说明
import tensorflow as tf

g = tf.Graph()
with g.as_default():
    data1 = tf.constant(2.5) # 常量
    data1_1 = tf.constant(2, dtype=tf.int32) 
    data2 = tf.Variable(10,name='var') # 变量

    init_op = tf.compat.v1.global_variables_initializer()
    sess = tf.compat.v1.Session(graph=g)
    with sess:
        # 变量需要初始化   
        sess.run(init_op)
    
        print(sess.run(data1))
        print(sess.run(data1_1))
        print(sess.run(data2))

        # 本质 tf = tensor+ 计算图
        # tensor 数据
        # op
        # graphs 数据操作
        # session
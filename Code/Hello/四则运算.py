# 常用数据类型的说明
import tensorflow as tf

g = tf.Graph()
with g.as_default():
    data1 = tf.constant(6)
    data2 = tf.Variable(2)

    dataAdd = tf.add(data1, data2)
    dataCopy = tf.compat.v1.assign(data2, dataAdd)
    dataMul = tf.multiply(data1, data2)
    dataSub = tf.subtract(data1, data2)
    dataDiv = tf.divide(data1, data2)

    init_op = tf.compat.v1.global_variables_initializer()
    sess = tf.compat.v1.Session(graph=g)
    with sess:
        sess.run(init_op)
        print(sess.run(dataAdd))
        print(sess.run(dataMul))
        print(sess.run(dataSub))
        print(sess.run(dataDiv))

        # 8->data2
        print('sess.run(dataCopy) ', sess.run(dataCopy))
        # 8+6->14->data=14
        print('dataCopy.eval() ', dataCopy.eval())
        # 14+6->20->data=20
        print('tf.get_default_session().run(dataCopy) ', 
            tf.compat.v1.get_default_session().run(dataCopy))

    print("end!")
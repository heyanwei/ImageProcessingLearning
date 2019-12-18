import tensorflow as tf


g = tf.Graph()
with g.as_default():
    data1 = tf.compat.v1.placeholder(tf.float32)
    data2 = tf.compat.v1.placeholder(tf.float32)

    mat0 = tf.constant([[0,0,0],[0,0,0]])
    mat1 = tf.zeros([2,3]) # 用零填充2行3列
    mat2 = tf.ones([3,2]) # 用一填充3行两列
    mat3 = tf.fill([2,3], 15) # 用15填充2行3列
    mat4 = tf.zeros_like(mat1) # 像矩阵1一样填充
    mat5 = tf.linspace(0.0,2.0,11) # 用11份把0到2等分
    mat6 = tf.compat.v1.random_uniform([2,3], -1,2) # 在2*3的矩阵填充-1到2的随机数

    data3 = tf.constant([[6,6]]) # [[6,6]]一行两列 
    data4 = tf.constant([[2],[2]]) # [[2],[2]] 两行一列
    data5 = tf.constant([[3,3]])
    data6 = tf.constant([[1,2],[3,4],[5,6]]) # [[1,2],[3,4],[5,6]] 三行两列

    print(data6.shape) # 维度 (3, 2)

    dataAdd = tf.add(data1,data2)
    dataMul = tf.matmul(data3, data4) # 矩阵相乘

    with tf.compat.v1.Session() as sess:
        print(sess.run(dataAdd, feed_dict={data1:6, data2:2}))
        
        print(sess.run(data6)) # 打印整体内容
        print(sess.run(data6[0])) # 打印某一行 [1 2]
        print(sess.run(data6[:,0])) # 打印某一列 [1 3 5]
        print(sess.run(data6[0,1])) # 打印第一行第二列 2
        print("dataMul: ", sess.run(dataMul))
        print("mat5: ", sess.run(mat5))
        print("mat6: ", sess.run(mat6))

print("end!")

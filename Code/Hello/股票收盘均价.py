import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

date = np.linspace(1,15,15)
endPrice = np.array([2511.90,2538.26,2510.68,
    2591.66,2732.98,2701.69,2701.29,2678.67,
    2726.50,2681.50,2739.17,2715.07,2823.58,
    2864.90,2919.68])
beginPrice =np.array([2438.70,2500.88,2534.95,
    2512.52,2594.04,2743.26,2697.47,2695.24,
    2678.23,2722.13,2674.93,2744.13,2717.46,
    2832.73,2877.40])

print(date)
plt.figure() # 绘图

for i in range(0,15):
    # 1 柱状图
    dateOne = np.zeros([2])
    dateOne[0] = i; # 开盘时间
    dateOne[1] = i; # 收盘时间
    priceOne = np.zeros([2])
    priceOne[0] = beginPrice[i] # 开盘价格
    priceOne[1] = endPrice[i] # 收盘价格
    if endPrice[i]>beginPrice[i]:
        plt.plot(dateOne, priceOne, 'r', lw = 8)
    else:
        plt.plot(dateOne, priceOne, 'g', lw = 8)

# plt.show()

# A(15*1)*w1(1*10)+b1(1*10)=B(15*10)
# B(15*10)*w2(10*1)+b2(15*1)=C(15*1)

dateNormal = np.zeros([15, 1])
priceNormal = np.zeros([15, 1])
for i in range(0, 15):
    dateNormal[i,0] = i/14.0; # 日期从0开始，最大14
    priceNormal[i,0] = endPrice[i]/3000.0; # 价格最大值不会超过3000

g = tf.Graph()
with g.as_default():
    x = tf.compat.v1.placeholder(tf.float32, [None,1]) # n行1列
    y = tf.compat.v1.placeholder(tf.float32, [None,1])

    # 隐藏层B
    w1 = tf.compat.v1.Variable(tf.compat.v1.random_uniform([1,10],0,1))
    b1 = tf.compat.v1.Variable(tf.zeros([1,10]))
    wb1 = tf.matmul(x,w1)+b1 # wb1 = x*w1+b1
    layer1 = tf.nn.relu(wb1) # 激励函数

    # 输出层C
    w2 = tf.compat.v1.Variable(tf.compat.v1.random_uniform([10,1],0,1))
    b2 = tf.compat.v1.Variable(tf.zeros([15,1]))
    wb2 = tf.matmul(layer1,w2)+b2
    layer2 = tf.nn.relu(wb2)
    loss = tf.reduce_mean(tf.square(y-layer2)) # y 真实 layer2 计算
    train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        for i in range(0,10000):
            sess.run(train_step,feed_dict={x:dateNormal, y:priceNormal})
        pred = sess.run(layer2,feed_dict={x:dateNormal})
        predPrice = np.zeros([15,1])
        for i in range(0,15):
            predPrice[i,0]=(pred*3000)[i,0]
        plt.plot(date,predPrice, 'b', lw = 1)

plt.show()

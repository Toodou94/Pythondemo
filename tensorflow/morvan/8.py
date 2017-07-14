# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 13:59:10 2017

@author: zhangli
"""

import tensorflow as tf
import numpy as np

#create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

###create tensorflow structure start 开始创建结构###
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))#定义生成一个随机的从-1~1的数
biases = tf.Variable(tf.zeros([1]))#定义初始值为0

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))#计算预测的y和实际的y的差别
optimizer = tf.train.GradientDescentOptimizer(0.5)#建立一个优化器，用这个优化器减小神经网络的误差提高准确度，0.5是一个学习效率
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
###create tensorflow structure end 结束创建结构###

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(biases))
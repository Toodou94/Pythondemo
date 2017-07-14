# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 14:15:20 2017

@author: zhangli
"""

import tensorflow as tf

matrix1 = tf.constant([[3,3]])#1行2列的矩阵
matrix2 = tf.constant([[2],
                       [2]])

product = tf.matmul(matrix1,matrix2) 

#method 1
sess = tf.Session() #定义Session
result = sess.run(product)
print(result)
sess.close()


#method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)
    
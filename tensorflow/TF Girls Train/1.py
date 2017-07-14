# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 14:01:12 2017

@author: zhangli
"""

import tensorflow as tf

#V1
#打印当前tensorflow版本
print('Loaded TF version',tf.__version__)


#V2
#实现简单函数
def basic_operation():
    v1 = tf.Variable(10)
    v2 = tf.Variable(5)
    addv = v1 + v2
    print(addv)
    
    sess = tf.Session()
    tf.initialize_all_variables().run(session=sess)
    print('hello',sess.run(addv))
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 13:55:41 2017

@author: zhangli
"""

import numpy as np
import theano.tensor as T
import theano

#activation function example激励函数
x = T.dmatrix('x')
s = 1/(1+T.exp(-x))  #logistic or soft step,激励函数的一种
logistic = theano.function([x],s)
print(logistic([[0,1],[2,3]]))


# multiply outputs for a function
a,b = T.dmatrices('a','b')#定义两个同类型的function是dmatrices
diff= a - b
abs_diff = abs(diff)
diff_squared = diff**2
f = theano.function([a,b],[diff,abs_diff,diff_squared])
print(f(
        np.ones((2,2)),
        np.arange(4).reshape((2,2))))


#name for a function
x,y,w = T.dscalars('x','y','w')
z = (x+y)*w
f = theano.function([x,
              theano.In(y,value=1),
              theano.In(w,value=2,name='weights')],
              z)
print(f(23,2,weights=4))

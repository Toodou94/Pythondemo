# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import theano.tensor as T
from theano import function

#basic
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x,y],z)#input x,y,output z

print(f(6,3))

#to pretty-print the function

from theano import pp
print(pp(z))#看z是怎么组成的


#how about matrix 如何定义一个矩阵
x = T.dmatrix('x')#定义一个存量，d是float64，f是float32
y = T.dmatrix('y')
z = x + y#相加
f = function([x,y],z)

print(f(np.arange(12).reshape((3,4)),#x=三行四列的矩阵
        10*np.ones((3,4))))#y=三行四列全部为10的一个矩阵
#xy相加就是我们的矩阵
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 10:09:50 2017

@author: zhangli
"""

import numpy as np
import theano
import theano.tensor as T

state = theano.shared(np.array(0,dtype=np.float64),'state') #定义share
inc = T.scalar('inc',dtype=state.dtype)
accumulator = theano.function([inc],state,updates=[(state,state+inc)])


#to get variable value
print(state.get_value())
accumulator(1)
print(state.get_value())
accumulator(10)
print(state.get_value())


#to set variable value
state.set_value(-1)
accumulator(3)
print(state.get_value())


#temporarily replace shared variable with another value in another function临时的方程，share value但是不更新
tmp_func = state*2 + inc
a = T.scalar(dtype=state.dtype)
skip_shared = theano.function([inc,a],tmp_func,givens=[(state,a)])#given=代替
print(skip_shared(2,3))
print(state.get_value())
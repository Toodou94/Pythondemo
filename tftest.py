# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 11:31:29 2017

@author: zhangli
"""

import matplotlib.pyplot as plt
import numpy as np
 
gold,chihh = 250,200

gold_height = 40 + 10* np.random.randn(gold)
chihh_height = 25 + 6* np.random.randn(chihh)

plt.hist([gold_height, chihh_height],stacked = True,color=['r','b'])
plt.show()
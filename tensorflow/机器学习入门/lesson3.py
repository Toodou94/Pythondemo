# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 09:50:04 2017

@author: zhangli
"""

import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

plt.hist([grey_height, lab_height],stacked=True, color=['r','b'])
plt.show()
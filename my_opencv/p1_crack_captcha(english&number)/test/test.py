"""
author: wiken
Date:2019/6/2
"""
import numpy as np

l = np.zeros((1, 3))
s = [2,2,2]
s1 = [3,3,3]
l = np.vstack((l, s))
l = np.vstack((l, s1))
print(l)
print("_____________________")
print(np.delete(l, 0,  axis=0))











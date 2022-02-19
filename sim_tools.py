# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 18:53:36 2021

@author: wes_c
"""

import math
import statistics as st
import numpy.random as random
from itertools import combinations

def biased_coin(p):
    if random.uniform() < p:
        return True
    else:
        return False
        
def cascade_segments(arr):
        # Sorting for efficiency
        arr.sort(key = lambda x: x[0])
        m = []
        s = -100000000
        max_ = -100000000
        for i in range(len(arr)):
            a = arr[i]
            if a[0] > max_:
                if i != 0:
                    m.append([s, max_])
                max_ = a[1]
                s = a[0]
            else:
                if a[1] >= max_:
                    max_ = a[1]
        #'max' value gives the last point of
        # that particular interval
        # 's' gives the starting point of that interval
        if max_ != -100000 and [s, max_] not in m:
            m.append([s, max_])
        return m
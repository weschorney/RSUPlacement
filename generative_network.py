# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 13:13:22 2021

@author: wes_c
"""

import tensorflow as tf
import tensorflow.keras as tk

#input is flattened road info (len, x, y, sp_lim, sp_stdev, orient (1-hot))
#output is 2*n RSU positions in form of x1,y1, ..., xn, yn
#TODO: Normalize input and unscale output?

class RSUPlacementGenerator(tf.keras.Model):
    def __init__(self, n_rsu):
        super(RSUPlacementGenerator, self).__init__()
        self.input_layer = tk.layers.InputLayer()
        self.first_conv = tk.layers.Conv1D()
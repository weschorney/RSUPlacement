# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 14:48:18 2021

@author: wes_c
"""

import tensorflow as tf
import tensorflow.keras as tk

#each road contributes 6 elements to input vector
#TODO: Normalize? Currently linear

class FullyConnectedNetwork(tk.Model):
    def __init__(self, max_roads, n_rsu, middle_layers=4,
                 dim_middle_layers=None):
        super(tk.Model, self).__init__()
        self.x1 = tk.layers.Dense(6*max_roads, input_shape=(None, 6*max_roads),
                                  activation='relu')
        if dim_middle_layers == None:
            self.dim_middle_layers = 4*max_roads
        else:
            self.dim_middle_layers = dim_middle_layers
        self.middle_layers = [tk.layers.Dense(self.dim_middle_layers,
                                              activation='relu')\
                              for _ in range(middle_layers)]
        self.final = tk.layers.Dense(2*n_rsu, activation='tanh')

    def call(self, inp):
        out = self.x1(inp)
        for layer in self.middle_layers:
            out = layer(out)
        final = self.final(out)
        return final

    
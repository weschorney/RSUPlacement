# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 18:53:16 2021

@author: wes_c
"""

import numpy as np

#inputs here are vectors coming from road posn's

class DenseLayer:
    def __init__(self, dim, bias=True):
        self.dim = dim
        self.has_bias = bias
        if bias:
            self.bias = np.random.random(size=(dim[0], 1))
        else:
            self.bias = np.zeros((dim[0], 1))
        self.mat = np.random.random(size=dim)

    def out(self, inp):
        return self.mat.dot(inp) + self.bias

    def update(self, inp, inp2, simfunc, patience):
        #other layer is instantiated and compared to results from this
        #layer. If better results, move towards other layer. This update
        #is global; can look into single weight updates
        other_layer = DenseLayer(self.dim, bias=self.has_bias)
        selfout = self.out(inp)
        otherout = other_layer.out(inp)
        selfsim = simfunc(selfout, roads=inp2)
        othersim = simfunc(otherout, roads=inp2)
        if othersim < selfsim:
            #we want to maximize score
            return patience + 1
        else:
            #update relative to score increase
            const = min(1, (othersim - selfsim)/othersim)
            self.mat = const*other_layer.mat + (1 - const)*self.mat
            return patience

class DenseNetwork:
    def __init__(self, layers, inp_dim, out_dim, bias=True, patience=25):
        self.layers = [DenseLayer(inp_dim, bias=bias) for _ in range(1, layers)]
        self.out_layer = DenseLayer(out_dim, bias=bias) #no out bias for now
        self.patience = patience

    def fit(self, infunc, simfunc, *args, n_rounds=100):
        for rnd in range(n_rounds):
            print(f"Beginning training round {rnd} of {n_rounds}.")
        #infunc gives an input vector to use with simfunc
            inp, inp2 = infunc(*args)
            for idx, _ in enumerate(self.layers):
                #optimize given layer while patience
                temp_inp = inp
                current_patience = 0
                while current_patience < self.patience:
                    for j in range(idx + 1):
                        if j != idx:
                            #just get input
                            temp_inp = self.layers[j].out(temp_inp)
                        else:
                            #this is layer we optimize
                            current_patience = self.layers[j].update(temp_inp,
                                                                     inp2,
                                                                     simfunc,
                                                                     self.patience)
            #finally optimize last layer
            for layer in self.layers:
                inp = layer.out(inp)
            self.out_layer.update(inp, inp2, simfunc, self.patience)
        return

    def out(self, inp):
        for layer in self.layers:
            inp = layer.out(inp)
        return self.out_layer.out(inp)

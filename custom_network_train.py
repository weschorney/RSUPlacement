# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 19:32:22 2021

@author: wes_c
"""

import numpy as np

from simulation_creator import get_roads
from road_system import RoadSystem
from custom_layers import DenseNetwork

def norm_road_inp(road_tup):
    args = road_tup[0]
    kwargs = road_tup[1]
    normed = np.zeros((6, )) #ignore stdev since doesn't change
    normed[0] = args[0]/750
    normed[1] = args[1]/850
    normed[2] = args[2]/850
    normed[3] = (args[3] - 20)/5
    normed[4] = 1 if args[5] == 'vertical' else 0
    normed[5] = (kwargs['start_lmbd'] - 5)/15
    return normed

def generate_inputs(n_roads):
    roads = get_roads(n_roads)
    #start, end lambda are same
    #(length, start_x, start_y, speed_limit, speed_limit_sigma, orient)
    normed_inputs = np.zeros(shape=(n_roads, 6))
    for idx, road in enumerate(roads):
        normed_inputs[idx, :] = norm_road_inp(road)
    normed_inputs = normed_inputs.reshape(6*n_roads,)
    return normed_inputs, roads

def obj_simulation(rsu_locations, roads=[], time=10000, rsu_range=250,
                   budget=1500, rsu_cost=250):
    #RSU Locations is list of len 2n
    rsus = []
    for i in range(0, len(rsu_locations), 2):
        rsus.append((rsu_locations[i], rsu_locations[i+1]))
    sim = RoadSystem(rsus, rsu_range, time, budget=budget, rsu_cost=rsu_cost)
    for arg, kwarg in roads:
        sim.add_road(*arg, **kwarg)
    score = sim.simulate()
    return score

def train_model(n_roads, layers=3):
    model = DenseNetwork(layers, (6, 60), 6)
    model.fit(generate_inputs, obj_simulation)
    #TODO: DEBUG?
    return model

if __name__ == '__main__':
    m = train_model(10)
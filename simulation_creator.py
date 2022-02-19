# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 17:05:22 2021

@author: wes_c
"""

import cma
import numpy.random as random
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from copy import copy
from functools import partial
from bayes_opt import BayesianOptimization
from road_system import RoadSystem

def create_random_road(min_road_length=150, max_road_length=750,
                       sim_x=1000, sim_y=1000, sp_lim_mu=20,
                       sp_lim_stdev=5, stdev_mu=lambda x: x/5,
                       lambda_min=5, lambda_max=20):
    orient = random.choice(['horizontal', 'vertical'])
    length = random.uniform(min_road_length, max_road_length)
    if orient == 'horizontal':
        start_x = random.uniform(0, sim_x - length)
        start_y = random.uniform(0, sim_y)
    else:
        start_x = random.uniform(0, sim_x)
        start_y = random.uniform(0, sim_y - length)
    speed_limit = random.normal(sp_lim_mu, sp_lim_stdev)
    speed_limit_sigma = stdev_mu(speed_limit)
    road_lambda = random.uniform(lambda_min, lambda_max)
    args = (length, start_x, start_y, speed_limit, speed_limit_sigma, orient)
    kwargs = {'start_lmbd':road_lambda, 'end_lmbd':road_lambda}
    return args, kwargs
    

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
    return -score

def bayes_obj_sim(roads=[], time=10000, rsu_range=250, rsu_cost=250,
                  budget=1500, **kwargs):
    rsus = []
    for i in range(len(kwargs)//2):
        rsus.append((kwargs.get('x' + str(i)), kwargs.get('y' + str(i))))
    sim = RoadSystem(rsus, rsu_range, time, budget=budget, rsu_cost=rsu_cost)
    for arg, kwarg in roads:
        sim.add_road(*arg, **kwarg)
    score = sim.simulate()
    return score

def get_roads(n_roads, **kwargs):
    roads = []
    for _ in range(n_roads):
        arg, kwarg = create_random_road(**kwargs)
        roads.append((arg, kwarg))
    return roads

def make_objective_function(roads, time, rsu_range, budget, rsu_cost):
    return partial(obj_simulation, roads=roads, time=time, rsu_cost=rsu_cost,
                   rsu_range=rsu_range, budget=budget)

def make_bayes_obj_fn(roads, time, rsu_range, rsu_cost, budget):
    return partial(bayes_obj_sim, roads=roads, time=time, rsu_range=rsu_range,
                   rsu_cost=rsu_cost, budget=budget)

def optimize(n_rsu, n_roads, time, rsu_range, rsu_cost, budget, **kwargs):
    es = cma.CMAEvolutionStrategy([0, 0] * n_rsu, 0.5)
    roads = get_roads(n_roads, **kwargs)
    opt_func = make_objective_function(roads, time, rsu_range,
                                       budget, rsu_cost)
    es.optimize(opt_func)
    return es

def bayes_optimize(n_rsu, n_roads, time, rsu_range,
                   rsu_cost, budget, init_points=20, n_iter=250, **kwargs):
    x_rng = kwargs.get('sim_x', 1000)
    y_rng = kwargs.get('sim_y', 1000)
    roads = get_roads(n_roads, **kwargs)
    pbounds = {}
    for i in range(n_rsu):
        pbounds['x' + str(i)] = (0, x_rng)
        pbounds['y' + str(i)] = (0, y_rng)
    opt_func = make_bayes_obj_fn(roads, time, rsu_range, rsu_cost, budget)
    optimizer = BayesianOptimization(
                    f=opt_func,
                    pbounds=pbounds,
                    random_state=777,
                    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    return optimizer, roads

def bayes_history_plot(opt, save=None):
    target = []
    iteration = []
    for idx, d in enumerate(opt.res):
        target.append(d['target'])
        iteration.append(idx)
    #normalize target
    target = [(ele - min(target))/(max(target) - min(target))\
              for ele in target]
    plt.plot(iteration, target)
    plt.xlabel('Iteration Number')
    plt.ylabel('Score Value (Normalized)')
    plt.title('Optimization Results')
    if isinstance(save, str):
        plt.savefig(save, dpi=500)
    plt.clf()
    return

def recreate_road(road):
    start = (road[0][1], road[0][2])
    if road[0][-1] == 'vertical':
        end = (start[0], start[1] + road[0][0])
    else:
        end = (start[0] + road[0][0], start[1])
    return [start, end]

def simulation_state_plot(opt, roads, rsu_range=250,
                          x_max=1000, y_max=1000, save=None):
    if not isinstance(opt, dict):
        opt = opt.max
    params = opt['params']
    #make rsu locations
    rsus = []
    for i in range(len(params)//2):
        rsus.append((params['x' + str(i)], params['y' + str(i)]))
    roads = [recreate_road(road) for road in roads]
    road_collection = mc.LineCollection(roads, linewidths=2)
    fig, ax = plt.subplots()
    ax.set_xlim([0, x_max])
    ax.set_ylim([0, y_max])
    ax.set_aspect('equal')
    ax.add_collection(road_collection)
    for xy in rsus:
        circle = plt.Circle(xy, rsu_range, alpha=0.3)
        new_circle = copy(circle)
        ax.add_artist(new_circle)
    ax.set_title('RSU Placement')
    plt.show()
    if isinstance(save, str):
        fig.savefig(save, dpi=500)
    return

def full_bayes_sim(min_rsu, max_rsu, n_roads, time, rsu_range,
                   rsu_cost, budget, init_points=20, n_iter=250,  **kwargs):
    x_rng = kwargs.get('sim_x', 1000)
    y_rng = kwargs.get('sim_y', 1000)
    roads = get_roads(n_roads, **kwargs)
    best = {'target':-1000000}
    #make bayesian bounds
    for i in range(min_rsu, max_rsu + 1):
        pbounds = {}
        for j in range(i):
            pbounds['x' + str(j)] = (0, x_rng)
            pbounds['y' + str(j)] = (0, y_rng)
        opt_func = make_bayes_obj_fn(roads, time, rsu_range, rsu_cost, budget)
        optimizer = BayesianOptimization(
                    f=opt_func,
                    pbounds=pbounds,
                    random_state=777,
                    )
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        if optimizer.max['target'] > best['target']:
            best = optimizer.max
    return best, optimizer, roads

if __name__ == '__main__':
#    es = optimize(5, 12, 10000, 250, 250, 1500)
#    opt, roads = bayes_optimize(9, 20, 10000, 100, 250, 2250, n_iter=800,
#                                init_points=200)
#    bayes_history_plot(opt, save='sim_results.png')
#    simulation_state_plot(opt, roads, rsu_range=100, save='test2.png')
    best, opt, roads = full_bayes_sim(3, 5, 10, 1000, 100, 200, 1000,
                                      init_points=4, n_iter=5)

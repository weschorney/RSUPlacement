# -*- coding: utf-8 -*-
"""
Created on Wed May  5 17:58:14 2021

@author: wes_c
"""

from simulation_creator import full_bayes_sim

def sort_output(best_pairs):
    #return in form x0, y0, x1, y1, ...
    pairs = best_pairs['params']
    out = []
    for i in range(len(pairs.keys())//2):
        out.append(pairs['x' + str(i)])
        out.append(pairs['y' + str(i)])
    return out

def roads_flatten(roads):
    out = []
    for road in roads:
        rd = list(road[0])
        rd[-1] = 1 if rd[-1] == 'horizontal' else 0
        flattened = [ele for ele in rd]
        flattened += list(road[1].values())
        out.append(flattened)
    out = [x for y in out for x in y]
    return out

def write_corpus(in_, name):
    with open(name, 'w') as f:
        for ele in in_:
            f.write(str(ele) + '\n')
    return

def create_dataset(runs=10000, in_corpus='roads.txt',
                   target_corpus='positions.txt'):
    targets = []
    roads = []
    for i in range(runs):
        best, _, rds = full_bayes_sim(3, 6, 10, 750, 100, 200, 1200,
                                        init_points=50, n_iter=150)
        targets.append(sort_output(best))
        roads.append(roads_flatten(rds))
        if i % 250 == 0:
            print(f'Finished iteration {i} out of {runs}')
            write_corpus(roads, in_corpus)
            write_corpus(targets, target_corpus)
    write_corpus(roads, in_corpus)
    write_corpus(targets, target_corpus)
    return

if __name__ == '__main__':
    create_dataset()

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 18:45:07 2021

@author: wes_c
"""

import numpy.random as random
from uuid import uuid1
from car import Car

class StraightRoad:
    def __init__(self, length, start_x, start_y, speed_limit, sp_lim_stdev,
                 orient='vertical', start_lmbd=10, end_lmbd=10):
        self.id_ = uuid1().hex
        self.length = length
        self.start_x = start_x
        self.start_y = start_y
        self.orient = orient
        self.start_lmbd = start_lmbd
        self.end_lmbd = end_lmbd
        self.speed_limit = speed_limit
        self.sp_lim_stdev = sp_lim_stdev
        self.coverage = None
        if orient == 'vertical':
            self.end_y = self.start_y + length
            self.end_x = self.start_x
        elif orient == 'horizontal':
            self.end_x = self.start_x + length
            self.end_y = self.start_y
        else:
            raise ValueError(f"orient must be horizontal or vertical, got {orient}")

    def is_in(self, point):
        if (point[0] >= self.start_x and point[0] <= self.end_x)\
        and (point[1] >= self.start_y and point[1] <= self.end_y):
            return True
        else:
            return False

    def start_generation_times(self, max_time):
        times = []
        while sum(times) <= max_time:
            times.append(random.exponential(self.start_lmbd))
        times = [sum(times[:i]) for i in range(1, len(times))]
        times = times[:-1] #otherwise overflow
        return times

    def end_generation_times(self, max_time):
        times = []
        while sum(times) <= max_time:
            times.append(random.exponential(self.end_lmbd))
        times = [sum(times[:i]) for i in range(1, len(times))]
        times = times[:-1] #otherwise overflow
        return times

    def create_traffic(self, max_time):
        #all events in simulation look like ("type", time, other args)
        start_times = self.start_generation_times(max_time)
        end_times = self.end_generation_times(max_time)
        start_direction = 'right' if self.orient == 'horizontal' else 'up'
        end_direction = 'left' if self.orient == 'horizontal' else 'down'
        starts = [('generate', time,
                   random.normal(self.speed_limit, abs(self.sp_lim_stdev)),
                   start_direction, self.start_x, self.start_y,
                   self.id_) for time in start_times]
        ends = [('generate', time,
                   random.normal(self.speed_limit, self.sp_lim_stdev),
                   end_direction, self.end_x, self.end_y,
                   self.id_) for time in end_times]
        times = starts + ends
        times.sort(key=lambda x: x[1])
        return times

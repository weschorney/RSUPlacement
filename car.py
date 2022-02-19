# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 19:01:31 2021

@author: wes_c
"""

import math
import numpy as np
import sim_tools as st
from uuid import uuid1

class Car:
    def __init__(self, create_time, speed, direction, start_x, start_y,
                 road_id, ping_interval=5):
        self.speed = speed
        self.direction = direction
        self.next_direction = None
        self.on_road = road_id
        self.next_road = road_id
        #self.start_x = start_x
        #self.start_y = start_y
        self.x = start_x
        self.y = start_y
        self.next_x = None
        self.next_y = None
        self.next_speed = None
        self.id_ = uuid1().hex
        self.create_time = create_time
        self.last_update_time = create_time
        self.ping_interval = ping_interval

    def update_position(self, time):
        distance_travelled = self.speed * (time - self.last_update_time)
        if self.direction == 'left':
            self.x = self.x - distance_travelled
        elif self.direction == 'right':
            self.x = self.x + distance_travelled
        elif self.direction == 'up':
            self.y = self.y + distance_travelled
        else:
            self.y = self.y - distance_travelled
        #overwrite last updated time
        self.last_update_time = time
        return

    def get_future_position(self, time):
        #absolute
        distance_travelled = self.speed * (time - self.last_update_time)
        if self.direction == 'left':
            x = self.x - distance_travelled
            y = self.y
        elif self.direction == 'right':
            x = self.x + distance_travelled
            y = self.y
        elif self.direction == 'up':
            y = self.y + distance_travelled
            x = self.x
        else:
            y = self.y - distance_travelled
            x = self.x
        return x, y

    def next_ping_time(self):
        #return next ping time based on last_update_time
        return int((self.last_update_time // self.ping_interval) * self.ping_interval + self.ping_interval)

    def ping_times(self, max_time):
        #ping times have future position of vehicle
        ping_time = list(range(self.next_ping_time(), math.ceil(max_time) + 1,
                                self.ping_interval))
        times = [('ping', time, *self.get_future_position(time), self.on_road)\
                 for time in ping_time]
        self.last_update_time = max_time #update position since this is called up to turn
        self.direction = self.next_direction #update direction "at turn"
        self.x, self.y = self.next_x, self.next_y
        self.on_road = self.next_road
        if self.next_speed is not None:
            self.speed = self.next_speed
        return times

    def turn_decision(self):
        #Currently hard code straight/turn bias
        directions = {'up', 'left', 'down', 'right'}
        turns = list(directions - {self.direction})
        if st.biased_coin(0.8):
            self.next_direction = self.direction
            return
        else:
            self.next_direction = np.random.choice(turns)
            return

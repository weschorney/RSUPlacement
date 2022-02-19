# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 18:46:57 2021

@author: wes_c
"""

import math
import numpy as np
import numpy.random as random
import sim_tools as st
from itertools import combinations
from straight_road import StraightRoad
from car import Car

class RoadSystem:
    def __init__(self, rsu_locations, rsu_range, sim_time,
                 rsu_cost=250, budget=1000):
        self.cars = []
        self.roads = []
        self.intersections = []
        self.rsu_locations = rsu_locations
        self.rsu_range = rsu_range
        self.total_pings = 0
        self.covered_pings = 0
        self.sim_time = sim_time
        self.rsu_cost = rsu_cost
        self.budget = budget

    def add_road(self, length, start_x, start_y, speed_limit,
                 speed_lim_stdev, orient, start_lmbd=None, end_lmbd=None):
        if start_lmbd and end_lmbd:
            self.roads.append(StraightRoad(length, start_x, start_y,
                                           speed_limit, speed_lim_stdev,
                                           orient=orient, start_lmbd=start_lmbd,
                                           end_lmbd=end_lmbd))
        else:
            self.roads.append(StraightRoad(length, start_x, start_y, speed_limit,
                                           speed_lim_stdev, orient=orient))
        return

    def add_car(self, create_time, speed, direction, start_x, start_y,
                road_id):
        self.cars.append(Car(create_time, speed, direction, start_x, start_y,
                             road_id))
        return

    def add_car_random(self, speed):
        road = random.choice(self.roads)
        road_id = road.id_
        if random.randint(0, 2):
            #start from beginning
            start_x = road.start_x
            start_y = road.start_y
            if road.orient == 'horizontal':
                direction = 'right'
            else:
                direction = 'up'
            self.cars.append(Car(speed, direction, start_x, start_y, road_id))
        else:
            #start from end
            start_x = road.end_x
            start_y = road.end_y
            if road.orient == 'horizontal':
                direction = 'left'
            else:
                direction = 'down'
            self.cars.append(Car(speed, direction, start_x, start_y, road_id))
        return

    def get_road(self, id_):
        road = [road for road in self.roads if road.id_ == id_]
        return road[0]

    def get_car(self, id_):
        car = [car for car in self.cars if car.id_ == id_]
        return car[0]

    def get_intersect_areas(self):
        for road1, road2 in combinations(self.roads, 2):
            if road1.orient == road2.orient:
                pass
            if road1.orient == 'horizontal':
                road1, road2 = road2, road1 #let road1 be vertical
            int_point = (road1.start_x, road2.start_y)
            if road1.is_in(int_point) and road2.is_in(int_point):
                self.intersections.append(int_point)
        return

    def get_points_of_interest(self, car):
        #direction is left, right, up, down
        #return all possible points car needs to make decision or be
        #destroyed in road system
        x = car.x
        y = car.y
        direction = car.direction
        #get intersections
        if direction == 'left':
            interest_points = [('turn', *pt) for pt in self.intersections\
                               if pt[1] == y and pt[0] < x]
        elif direction == 'right':
            interest_points = [('turn', *pt) for pt in self.intersections\
                               if pt[1] == y and pt[0] > x]
        elif direction == 'up':
            interest_points = [('turn', *pt) for pt in self.intersections\
                               if pt[0] == x and pt[1] > y]
        else:
            interest_points = [('turn', *pt) for pt in self.intersections\
                               if pt[0] == x and pt[1] < y]
        #add end of current road
        road = self.get_road(car.on_road)
        if direction == 'right' or direction == 'up':
            interest_points.append(('destroy', road.end_x, road.end_y))
        else:
            interest_points.append(('destroy', road.start_x, road.start_y))
        return interest_points

    def _road_rsu_coverage(self, rsu, pts):
        #RSU is just a point
        covered_pts = [pt for pt in pts if \
            np.linalg.norm([pt[0] - rsu[0], pt[1] - rsu[1]]) <= self.rsu_range]
        try:
            m = min(covered_pts)
            M = max(covered_pts)
        except:
            m = (1, 1)
            M = (1, 1) #arbitrarily choose point with approx 0 probability it give false signal
        covered_range = [m, M]
        return covered_range

    def _get_road_coverage(self, road, granularity=0.2):
        if road.orient == 'horizontal':
            pts = np.linspace(road.start_x, road.end_x,
                        num=math.ceil((road.end_x - road.start_x)/granularity))
            pts[-1] = pts[-1] if pts[-1] < road.end_x else road.end_x
            pts = list(zip(pts, [road.start_y] * len(pts)))
            covered_ranges = [self._road_rsu_coverage(rsu, pts)\
                              for rsu in self.rsu_locations]
            #just x values of ranges
            covered_ranges = [[rg[0][0], rg[1][0]] for rg in covered_ranges]
            covered = st.cascade_segments(covered_ranges)
            #DO NOT add back in the y-values
            road.coverage = covered
        else:
            pts = np.linspace(road.start_y, road.end_y,
                        num=math.ceil((road.end_y - road.start_y)/granularity))
            pts[-1] = pts[-1] if pts[-1] < road.end_y else road.end_y
            pts = list(zip(pts, [road.start_x] * len(pts)))
            covered_ranges = [self._road_rsu_coverage(rsu, pts)\
                              for rsu in self.rsu_locations]
            #just y values of ranges
            covered_ranges = [[rg[0][1], rg[1][1]] for rg in covered_ranges]
            covered = st.cascade_segments(covered_ranges)
            #DO NOT add back in the x-values
            road.coverage = covered
        return

    def calculate_road_coverage(self, granularity=0.2):
        for road in self.roads:
            self._get_road_coverage(road, granularity=granularity)
        return

    def _get_closest_intersection(self, x, y, int_list, road_orientation,
                                  car_direction):
        if road_orientation == 'horizontal':
            if car_direction == 'left':
                int_list = [ele for ele in int_list if ele[0] < x]
            else:
                int_list = [ele for ele in int_list if ele[0] > x]
            int_list.sort(key=lambda q: abs(q[0] - x))
            return int_list[0] if int_list else []
        elif road_orientation == 'vertical':
            if car_direction == 'down':
                int_list = [ele for ele in int_list if ele[1] < y]
            else:
                int_list = [ele for ele in int_list if ele[1] > y]
            int_list.sort(key=lambda q: abs(q[1] - y))
            return int_list[0] if int_list else []
        else:
            raise ValueError(f"road_orientation must be horizontal or vertical, got {road_orientation}")

    def _get_intersection(self, car):
        #events are pings and turns/sinks 
        #need to recalculate events and update "start pos'n" every time
        #a turn is made. since we recalculate and update after every
        #turn point (even if continue straight), only need events up to
        #the turn point, after which we query for new ones.
        #first, we get the intersection
        road = self.get_road(car.on_road)
        if road.orient == 'horizontal':
            next_intersection = [int_ for int_ in self.intersections if\
                                 int_[1] == car.y] #have to be on same y-val
            try:
                next_intersection = self._get_closest_intersection(car.x, car.y,
                                                                   next_intersection,
                                                                   road.orient,
                                                                   car.direction)
            except:
                pass
            if not next_intersection:
                #get road end; car will be destroyed
                if car.direction == 'left':
                    next_intersection = (road.start_x, road.start_y)
                else:
                    next_intersection = (road.end_x, road.end_y)
                #get time to this point, add last update time since car will
                #update at every turn and on creation
                t = abs(next_intersection[0] - car.x)/car.speed + car.last_update_time
                return ('destroy', t, car.id_)
            #get distance to next intersection
            t = abs(next_intersection[0] - car.x)/car.speed + car.last_update_time
            #return turn decision and update car position and direction
            car.next_x, car.next_y = next_intersection
            car.turn_decision()
            if car.next_direction == 'up' or car.next_direction == 'down':
                #car is now on new road, need to update id_
                new_road = [road for road in self.roads if road.start_x == car.next_x\
                            and road.orient == 'vertical'][0] #debug this make sure 0
                car.next_road = new_road.id_
                car.next_speed = random.normal(new_road.speed_limit, new_road.sp_lim_stdev)
            #car.last_update_time = t need to set this after ping times
            return ('turn', t, next_intersection[0], next_intersection[1],
                    car.id_) #return road intersection coords in case rounding errors
        else:
            next_intersection = [int_ for int_ in self.intersections if\
                                 int_[0] == car.x] #need to be on same x-val
            try:
                next_intersection = self._get_closest_intersection(car.x, car.y,
                                                                   next_intersection,
                                                                   road.orient,
                                                                   car.direction)
            except:
                pass
            if not next_intersection:
                #get road end; car will be destroyed
                if car.direction == 'down':
                    next_intersection = (road.start_x, road.start_y)
                else:
                    next_intersection = (road.end_x, road.end_y)
                t = abs(next_intersection[1] - car.y)/car.speed + car.last_update_time
                return ('destroy', t, car.id_)
            #get distance to next intersection
            t = abs(next_intersection[1] - car.y)/car.speed + car.last_update_time
            #return turn decision and update car position and direction
            car.next_x, car.next_y = next_intersection
            car.turn_decision()
            if car.next_direction == 'left' or car.next_direction == 'right':
                #car is on new road, update id_
                new_road = [road for road in self.roads if road.start_y == car.next_y\
                            and road.orient == 'horizontal'][0]
                car.next_road = new_road.id_
                car.next_speed = random.normal(new_road.speed_limit, new_road.sp_lim_stdev)
            #car.last_update_time = t need to set this after ping times
            return ('turn', t, next_intersection[0], next_intersection[1],
                    car.id_) #return intersection coords for rounding errors

    def get_event_stream(self, car):
        next_int = self._get_intersection(car)
        time_until = next_int[1] #time at turn
        events = car.ping_times(time_until)
        events = events + [next_int]
        return events

    def get_events(self, car):
        events = []
        while len(events) == 0 or events[-1][0] != 'destroy':
            events += self.get_event_stream(car)
        return events

    def get_all_events(self):
        all_events = []
        for car in self.cars:
            all_events.append(self.get_events(car))
        #flatten list and sort by time [2nd entry]
        all_events = [x for y in all_events for x in y]
        all_events.sort(key=lambda x: x[1])
        return all_events

    def _get_road_creation_events(self, road):
        events = road.create_traffic(self.sim_time)
        return events

    def calculate_all_creations(self):
        all_events = []
        for road in self.roads:
            all_events.append(self._get_road_creation_events(road))
        all_events = [x for y in all_events for x in y]
        all_events.sort(key=lambda x: x[1])
        return all_events

    ##################################################
    ########### SIMULATION EVENT HANDLERS ############
    ##################################################

    def _is_ping_covered(self, ping_event):
        road = self.get_road(ping_event[-1])
        if road.orient == 'horizontal':
            #check if x coordinate is covered
            if any(ping_event[2] >= cv[0] and ping_event[2] <= cv[1] for\
                   cv in road.coverage):
                self.covered_pings += 1
                self.total_pings += 1
            else:
                self.total_pings += 1
        elif road.orient == 'vertical':
            #check if y coordinate is covered
            if any(ping_event[3] >= cv[0] and ping_event[3] <= cv[1] for\
                   cv in road.coverage):
                self.covered_pings += 1
                self.total_pings += 1
            else:
                self.total_pings += 1
        return

    def _destroy_car(self, event):
        id_ = event[-1]
        self.cars = [cr for cr in self.cars if cr.id_ != id_]
        return

    def _create_car(self, event):
        info = event[1:]
        self.add_car(*info)
        return

    def event_stream_handler(self, events):
        for event in events:
            if event[0] == 'ping':
                self._is_ping_covered(event)
            elif event[0] == 'destroy':
                self._destroy_car(event)
            elif event[0] == 'generate':
                self._create_car(event)
            else:
                pass #nothing to do for a turn
        return

    ##################################################
    ########### SIMULATION SCORE HANDLERS ############
    ##################################################

    def _single_road_coverage(self, road):
        try:
            covered = sum(road.coverage)
            total = road.length
        except:
            covered = 0
            total = road.length
        return covered, total

    def road_coverage_score(self):
        covered = []
        total = []
        for road in self.roads:
            c, t = self._single_road_coverage(road)
            covered.append(c)
            total.append(t)
        return sum(covered)/sum(total)

    def ping_coverage_score(self):
        return self.covered_pings/self.total_pings

    def rsu_cost_score(self):
        return len(self.rsu_locations) * self.rsu_cost / self.budget

    def score(self):
        #weight pings more than raw coverage
        cov_score = self.road_coverage_score()
        ping_score = self.ping_coverage_score()
        cost_score = self.rsu_cost_score()
        full_score = 0.5*cov_score + 0.9*ping_score - 0.23*cost_score
        return full_score

    ##################################################
    ################## SIMULATION  ###################
    ##################################################

    def simulate(self, granularity=0.2):
        #first get intersections and road coverage
        self.get_intersect_areas()
        self.calculate_road_coverage(granularity=granularity)
        #next get all events
        #1 - generate traffic from each road
        creations = self.calculate_all_creations()
        #add cars
        self.event_stream_handler(creations)
        #next get all car events
        events = self.get_all_events()
        self.event_stream_handler(events)
        score = self.score()
        return score

if __name__ == '__main__':
    ############## TEST ONE ###############
    sim = RoadSystem([(0,0), (10, 10), (10, -10)], 250, 10000)
    #two crossroads
    sim.add_road(1000, 5, 75, 50, 15, 'vertical')
    sim.add_road(2000, -75, 100, 75, 20, 'horizontal')
    score = sim.simulate()
    print(score)
    ############## TEST TWO ###############
#    sim = RoadSystem([(0,0), (10, 10), (10, -10)], 250, 10000)
#    sim.add_road(100, 5, 75, 50, 15, 'vertical')
#    sim.add_road(200, -75, 100, 75, 20, 'horizontal')
#    sim.add_car(0, 1, 'up', 5, 75, sim.roads[0].id_)
#    sim.get_intersect_areas()
#    sim.calculate_road_coverage(granularity=0.2)
#    e = sim.get_all_events()
#    print(e)

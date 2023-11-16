#!/usr/bin/python3

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
# elif PYQT_VER == 'PYQT4':
# 	from PyQt4.QtCore import QLineF, QPointF
# elif PYQT_VER == 'PYQT6':
# 	from PyQt6.QtCore import QLineF, QPointF
# else:
# 	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


import time
import numpy as np
from TSPClasses import *
import heapq
import itertools


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        found_tour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not found_tour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                found_tour = True
        end_time = time.time()
        results['cost'] = bssf.cost if found_tour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

    def greedy(self, time_allowance=60.0):
        cities = self._scenario.getCities()
        ncities = len(cities)
        found_route = False
        count = 0
        bssf = None

        # start timer
        start_time = time.time()

        # try starting at each city
        for i in range(ncities):
            # check timer
            if (time.time() - start_time) >= time_allowance:
                # out of time
                break

            valid_route = True
            unvisited = set(cities)
            route = [cities[i]]

            # add the starting city
            curr_city = cities[i]
            unvisited.remove(cities[i])

            # add each city to the route
            while unvisited:
                closest_city_dist = np.inf
                closest_city = None

                # find the closest unvisited city
                for dest in unvisited:
                    assert dest is not curr_city
                    dist = curr_city.costTo(dest)
                    if dist < closest_city_dist:
                        closest_city_dist = dist
                        closest_city = dest

                if closest_city is None:
                    # no adjacent, unvisited city
                    valid_route = False
                    break

                # add the closest unvisited city to the route
                route.append(closest_city)
                unvisited.remove(closest_city)
                curr_city = closest_city

            if valid_route:
                count += 1
                # update the bssf if this route is better
                curr_solution = TSPSolution(route)
                if bssf is None or curr_solution.cost < bssf.cost:
                    # new best solution
                    found_route = True
                    bssf = curr_solution

        # stop timer
        end_time = time.time()

        print("ROUTE FOUND" if found_route else "NO ROUTE FOUND")
        return {'cost': bssf.cost if found_route else math.inf, 'time': end_time - start_time,
                'count': count, 'soln': bssf, 'max': None, 'total': None, 'pruned': None}

    ''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''

    def branchAndBound(self, time_allowance=60.0):
        pass

    ''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''
    def fancy(self, time_allowance=60.0):
        pass

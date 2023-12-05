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
        self.max_states = None
        # have numpy raise warnings as errors
        np.seterr(all='raise')

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
        # 1) INITIALIZATION
        cities = self._scenario.getCities()
        ncities = len(cities)
        self.max_states = 0
        count = 0
        # static variables for State
        State.ncities = ncities
        State.num_states_generated = 1  # initial state not created by expand function

        # start timer
        start_time = time.time()

        # create matrix for root state
        # | ∞  7  3  12 |
        # | 3  ∞  6  14 |
        # | 5  8  ∞  6  |
        # | 9  3  5  ∞  |
        root_matrix = np.zeros((ncities, ncities))
        for i in range(ncities):
            for j in range(ncities):
                if i == j:
                    # diagonal of infinities
                    root_matrix[i, j] = np.inf
                else:
                    # dist from row city to col city
                    root_matrix[i, j] = cities[i].costTo(cities[j])

        # create reduced-cost matrix of initial_matrix
        # | ∞  4  0  8  |
        # | 0  ∞  3  10 |
        # | 0  3  ∞  0  |
        # | 6  0  2  ∞  |
        State.initial_lower_bound = State.reduce_cost(root_matrix)

        # unvisited - include all cities except first (starting city)
        unvisited = [None] * (ncities - 1)
        city_to_index = {}
        for i in range(1, ncities):
            unvisited[i - 1] = cities[i]
            city_to_index[cities[i]] = i
        # cache indexes of each city
        State.city_to_index = city_to_index

        # create initial State
        initial_state = State(matrix=root_matrix, lower_bound=State.initial_lower_bound,
                              route=[cities[0]], unvisited=unvisited, from_row=0)
        priority = initial_state.get_priority()

        # priority queue to track states
        q = PriorityQueue()
        q.add_state(initial_state, priority)

        # 2) INITIAL BSSF - run greedy for 2 seconds max
        bssf = self.greedy(2 if time_allowance >= 2 else time_allowance)['soln']

        # 3) RUN ALGORITHM until timeout or no states left
        while not q.is_empty() and (time.time() - start_time) < time_allowance:
            # analyze a state
            state = q.eject_state()

            if bssf is None or state.lower_bound < bssf.cost:
                # potentially contains better route
                for child_state in state.expand():
                    if child_state.is_complete_route():
                        # route is complete
                        solution = TSPSolution(child_state.route)
                        if solution.cost != np.inf:
                            # infinite routes are not valid
                            if bssf is None or solution.cost < bssf.cost:
                                # route is better than previous BSSF
                                count += 1
                                bssf = solution
                    elif bssf is None or (child_state.lower_bound < bssf.cost and child_state.lower_bound != np.inf):
                        # route not yet complete AND potentially better than BSSF
                        # add it to the queue to analyze later
                        q.add_state(child_state, child_state.get_priority())
                    else:
                        # prune the state (don't put it on the q)
                        q.num_states_pruned += 1
            else:
                # prune the state (don't expand it)
                q.num_states_pruned += 1

        # stop timer
        end_time = time.time()

        return {'cost': bssf.cost if bssf is not None else math.inf, 'time': end_time - start_time,
                'count': count, 'soln': bssf, 'max': q.max_states, 'total': State.num_states_generated,
                'pruned': q.num_states_pruned}

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


class State:
    # static class variables
    # NOTE - these must be set before calling helper methods
    ncities = None
    initial_lower_bound = None
    num_states_generated = None
    city_to_index = None

    def __init__(self, matrix: np.ndarray, lower_bound: float, route: list, unvisited: list, from_row: int):
        self.matrix = matrix
        self.lower_bound = lower_bound
        self.route = route
        self.unvisited = unvisited
        self.from_row = from_row

    def get_priority(self) -> float:
        """
        Computes the priority for computing a given state.\n
        Prioritizes depth, as well as lower bound approximation.

        :return: The priority of the state, as a float
        """
        assert State.ncities is not None and State.initial_lower_bound
        PROGRESS_WEIGHT = 1
        LOWER_BOUND_WEIGHT = 0.5

        # CITIES_IN_ROUTE / TOTAL_CITIES
        progress_priority = (len(self.route) / State.ncities) * PROGRESS_WEIGHT

        # INITIAL_LOWER_BOUND / LOWER_BOUND
        cost_priority = (State.initial_lower_bound / self.lower_bound) * LOWER_BOUND_WEIGHT

        return (progress_priority + cost_priority) / 2

    def expand(self):
        child_states = []

        for dest in self.unvisited:
            # to_col is the index of the destination
            to_col = State.city_to_index[dest]

            # child's route
            child_route = self.route.copy()
            child_route.append(dest)

            # child's unvisited to hold all except its new dest
            child_unvisited = [None] * (len(self.unvisited) - 1)
            i = 0
            for city in self.unvisited:
                if city is not dest:
                    child_unvisited[i] = city
                    i += 1

            # copy and inf out respective parts of matrix
            child_matrix = self.matrix.copy()
            # set from_row to inf
            child_matrix[self.from_row, :] = np.inf
            # set to_col to inf
            child_matrix[:, to_col] = np.inf
            # set val at to_col, from_row to inf
            child_matrix[to_col, self.from_row] = np.inf

            # CHILD LOWER BOUND VALUE
            # 1) parent lower bound
            # 2) reduce cost matrix
            cost_to_reduce = State.reduce_cost(child_matrix)
            # 3) parent matrix at from_row, to_col
            parent_val = self.matrix.item(self.from_row, to_col)
            child_lower_bound = self.lower_bound + cost_to_reduce + parent_val

            # child's from_row
            child_from_row = to_col

            # add the child state to the list
            child_states.append(State(matrix=child_matrix, lower_bound=child_lower_bound, route=child_route,
                                      unvisited=child_unvisited, from_row=child_from_row))

        State.num_states_generated += len(child_states)

        return child_states

    def is_complete_route(self):
        # route is complete if contains every city
        return True if len(self.route) == State.ncities else False

    @staticmethod
    def reduce_cost(matrix: np.ndarray) -> float:
        """
        Subtracts the minimum row value from each row and the minimum column value from each column

        :param matrix: The numpy array to modify
        :return: The reduction cost, as a float
        """
        # reduce all rows
        row_mins = np.min(matrix, axis=1, keepdims=True)
        for i in range(len(row_mins)):
            if row_mins[i] == np.inf:
                # ignore infinity mins
                row_mins[i] = 0

        matrix -= row_mins

        # reduce cols
        col_mins = np.min(matrix, axis=0, keepdims=True)
        for i in range(len(col_mins[0])):
            if col_mins[0, i] == np.inf:
                # ignore infinity mins
                col_mins[0, i] = 0

        matrix -= col_mins

        return sum(row_mins)[0] + sum(col_mins[0])


class PriorityQueue:
    # static variable for item tiebreaker (FIFO)
    id = 0

    def __init__(self):
        self.max_states = 0
        self.num_states_pruned = 0
        self._heap = []

    def add_state(self, state: State, priority: float) -> None:
        """
        Adds another State to the PriorityQueue

        :param priority: The priority of the State to add, a float between 0 and 1
        :param state: The State to add to the PriorityQueue
        """
        # NOTE - negate priority to interpret as max heap
        # NOTE - we use PriorityQueue.id as the second argument for tiebreakers
        heapq.heappush(self._heap, (-priority, PriorityQueue.id, state))       # ]- O(logn)
        PriorityQueue.id += 1

        # update max_states
        self.max_states = max(self.max_states, len(self._heap))

    def eject_state(self) -> State:
        """
        Retrieves the State at the top of the PriorityQueue

        :return: The State at the top of the PriorityQueue
        """
        return heapq.heappop(self._heap)[2]                            # ]- O(logn)

    def is_empty(self):
        return len(self._heap) == 0

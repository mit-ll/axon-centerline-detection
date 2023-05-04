import numpy as np
import math
from abc import ABCMeta, abstractmethod
from heapq import heappush, heappop, heapify

class AStar(metaclass=ABCMeta):
    __slots__ = ('costs')

    def __init__(self, costs):
        self.costs = costs

    class SearchNode:
        __slots__ = ('data', 'gscore', 'fscore',
                     'closed', 'came_from', 'out_openset')

        def __init__(self, data, gscore=float('inf'), fscore=float('inf')):
            self.data = tuple(data)
            self.gscore = gscore
            self.fscore = fscore
            self.closed = False
            self.out_openset = True
            self.came_from = None

        def __lt__(self, b):
            return self.fscore < b.fscore

    class SearchNodeDict(dict):

        def __missing__(self, k):
            v = AStar.SearchNode(k)
            self.__setitem__(k, v)
            return v

    def heuristic_cost_estimate(self, current, goal):
        """Computes the estimated distance from the goal for each node in nodes. """
        return 0.5 * math.dist(current, goal)

    def distance_between(self, n1, n2):
        """Gives the real distance between two adjacent nodes n1 and n2 (i.e n2 belongs to the list of n1's neighbors).
           n2 is guaranteed to belong to the list returned by the call to neighbors(n1).
           This method must be implemented in a subclass."""
        return 1.1 - self.costs[n2]

    def neighbors(self, node):
        """For a given node, returns (or yields) the list of its neighbors. this method must be implemented in a subclass"""
        # List of coordinates in 3x3x3 cube
        neighbors = np.indices((3, 3, 3)).transpose(1, 2, 3, 0).reshape(-1, 3) - 1 + node
        # Filter out original point and anything out of image bounds
        neighbors = [n for n in neighbors
                     if not (np.array_equal(n, node)
                             or np.any((n < 0) | (n >= self.costs.shape))
                             )
                    ]
        return np.array(neighbors)

    def is_goal_reached(self, current, goal):
        """ returns true when we can consider that 'current' is the goal"""
        return np.array_equal(current, goal)

    def reconstruct_path(self, last, reversePath=False):
        def _gen():
            current = last
            while current:
                yield current.data
                current = current.came_from
        if reversePath:
            return _gen()
        else:
            return reversed(list(_gen()))

    def astar(self, start, goal, reversePath=False):
        if self.is_goal_reached(start, goal):
            return [start]
        searchNodes = AStar.SearchNodeDict()
        startNode = searchNodes[tuple(start)] = AStar.SearchNode(
            start, gscore=.0, fscore=self.heuristic_cost_estimate(start, goal))
        openSet = []
        heappush(openSet, startNode)
        while openSet:
            current = heappop(openSet)
            if self.is_goal_reached(current.data, goal):
                return self.reconstruct_path(current, reversePath)
            current.out_openset = True
            current.closed = True
            for neighbor in map(lambda n: searchNodes[tuple(n)], self.neighbors(current.data)):
                if neighbor.closed:
                    continue
                tentative_gscore = current.gscore + \
                    self.distance_between(current.data, neighbor.data)
                if tentative_gscore >= neighbor.gscore:
                    continue
                neighbor.came_from = current
                neighbor.gscore = tentative_gscore
                neighbor.fscore = tentative_gscore + \
                    self.heuristic_cost_estimate(neighbor.data, goal)
                if neighbor.out_openset:
                    neighbor.out_openset = False
                    heappush(openSet, neighbor)
                else:
                    # re-add the node in order to re-sort the heap
                    openSet.remove(neighbor)
                    heappush(openSet, neighbor)
        return None


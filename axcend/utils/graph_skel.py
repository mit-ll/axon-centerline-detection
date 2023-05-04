import math
import itertools
import numpy as np
from scipy.ndimage.filters import convolve
from concurrent.futures import ProcessPoolExecutor
from utils.astar import AStar
from utils.transforms import Skeletonize


class Graph:

    def __init__(self, segmentation, resolution=(1, 1, 1), delta=16, epsilon=8):
        self.segmentation = segmentation
        self.skeleton = Skeletonize(threshold=0.5).apply(self.segmentation)
        self.resolution = np.array(resolution)
        self.max_step_distance = np.sqrt((self.resolution**2).sum())
        self.delta = float(delta)
        self.epsilon = float(epsilon)
        print('getting nodes')
        self.nodes = self.get_nodes()
        print(len(self.nodes))
        self.paths = self.get_paths()

    def _get_neighbors(self, point):
        ''' Return list of coordinates of neighboring voxels. '''
        # List of coordinates in 3x3x3 cube
        neighbors = np.indices((3, 3, 3)).transpose(1, 2, 3, 0).reshape(-1, 3) - 1 + point
        # Filter out original point and anything out of image bounds
        neighbors = [n for n in neighbors
                     if not (np.array_equal(n, point)
                             or np.any((n < 0) | (n >= self.skeleton.shape))
                             )
                    ]
        return np.array(neighbors)

    def _get_distance(self, n1, n2):
        return math.dist(n1*self.resolution, n2*self.resolution)

    def _get_nodes(self, point, distance_traveled=0.):
        '''
        Recursively step through voxels of skeleton until forming a loop.
        Nodes are sampled at regular intervals as well as at every
        critical point (i.e. branching or endpoint).

        Args
        ----
            skel : skeletonized image
            point : (x, y, z) coordinates to evaluate
            distance_traveled : distance traveled since last node was placed

        Returns
        -------
            nodes : list of voxel coordinates representing nodes
        '''
        # Base case: we've formed a loop
        if point in self.visited:
            return []

        # Step forward
        self.visited.add(point)

        # Get neighboring voxels (within 3x3x3 cube) that lie along skeleton
        neighbors = self._get_neighbors(point)
        neighbors = neighbors[self.skeleton[tuple(neighbors.T)] == 1]

        nodes = []
        # Sample node every delta steps
        if distance_traveled >= self.delta:
            nodes.append(point)
            distance_traveled = 0.
        # Sample critical points, enforcing minimum step distance of epsilon
        elif distance_traveled > self.epsilon:
            if point in self.endpoints or len(neighbors) > 2:
                nodes.append(point)
                distance_traveled = 0.

        # Visit neighbors
        for neighbor in neighbors:
            distance_to_neighbor = self._get_distance(point, neighbor)
            nodes += self._get_nodes(tuple(neighbor), distance_traveled + distance_to_neighbor)

        return nodes

    def get_nodes(self):
        '''
        Sample nodes from skeletonized image, including all critical points.

        Returns
        -------
            nodes : numpy array of voxel coordinates representing nodes
        '''
        # Find endpoints; i.e., where there exists only 1 neighbor
        skel = np.pad(self.skeleton, (1,), mode='constant')
        kernel = np.ones((3, 3, 3))
        kernel[1, 1, 1] = 100
        self.endpoints = \
            set([tuple(i-1) for i in np.argwhere(convolve(skel, kernel, mode='constant') == 101)])

        # Recursively sample skeleton starting from each endpoint not yet visited
        nodes = []
        self.visited = set([])
        for point in self.endpoints:
            nodes += self._get_nodes(point, distance_traveled=self.delta)

        return np.array(nodes)

    def _get_path(self, pair):
        astar = AStar(self.segmentation)
        return astar.astar(*pair)

    def get_paths(self):
        '''
        Returns best path between every pair of connected nodes.
        '''
        assert len(self.nodes) > 1
        pairs = [pair for pair in itertools.combinations(self.nodes, 2)
                 if self._get_distance(*pair) <= self.delta + self.max_step_distance]

        paths = []
        with ProcessPoolExecutor() as executor:
            for path in executor.map(self._get_path, pairs, chunksize=128):
                paths.append(np.array(list(path)))

        return np.array(paths, dtype=object)

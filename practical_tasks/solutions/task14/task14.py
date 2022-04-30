import random
from collections import namedtuple
from typing import List, Callable

import numpy as np

Clusterisation = namedtuple('Clusterisation', ['points_to_clusters', 'centers', 'iterations'])


def clusterisation(points: List, dimension: int, k: int, dist: Callable) -> Clusterisation:
    centers, points_to_clusters, changed = random.sample(points, k), [-1 for _ in points], True
    iterations = 0
    while changed:
        changed = False
        for i, point in enumerate(points):
            n = min(enumerate(centers), key=lambda c: dist(c[1], point))[0]
            points_to_clusters[i], changed = n, points_to_clusters[i] != n
        for i in range(k):
            coordinates_sum, count = np.zeros(dimension), 0
            for cluster, point in zip(points_to_clusters, points):
                if cluster == i:
                    coordinates_sum += np.array(point)
                    count += 1
            centers[i] = coordinates_sum / count
        iterations += 1

    return Clusterisation(points_to_clusters, centers, iterations)

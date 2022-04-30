import os
import random
from math import sqrt
from typing import List, Callable

from matplotlib import pyplot as plt

from practical_tasks.solutions.task14.task14 import clusterisation


def experiment(points: List, k: int, dist: Callable, title: str, file_name: str):
    clusterisation_result = clusterisation(points, 2, k, dist)
    x, y = zip(*points)
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=clusterisation_result.points_to_clusters)

    fig.suptitle(f'{title}, {clusterisation_result.iterations}')
    if not os.path.isdir('results'):
        os.makedirs('results')
    fig.savefig(f'results/{file_name}.png')


def run():
    points = [(random.uniform(1, 30), random.uniform(1, 30)) for _ in range(500)]
    k = 5
    dists = [(lambda p1, p2: sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 'sqrt((x1-y1)^2+(x2-y2)^2)'),
             (lambda p1, p2: (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2, '(x1-y1)^2+(x2-y2)^2'),
             (lambda p1, p2: abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]), '|(x1-y1)|+|(x2-y2)|'),
             (lambda p1, p2: max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])), 'max(|(x1-y1)|,|(x2-y2)|)')]

    for i, dist in enumerate(dists, start=1):
        experiment(points, k, dist[0], dist[1], f'experiment{i}_1')
        experiment(points, k, dist[0], dist[1], f'experiment{i}_2')


if __name__ == '__main__':
    run()

import csv
import os
from typing import Iterable, List

from matplotlib import pyplot as plt


def create_table(path_to_file, file_name, titles, results):
    if not os.path.isdir(path_to_file):
        os.makedirs(path_to_file)

    with open(f'{path_to_file}/{file_name}', 'w') as f:
        write = csv.writer(f)
        write.writerow(titles)
        write.writerows(results)


def create_plot(path_to_file, file_name, x_vals: Iterable, each_graph_y_vals: List, main_title: str = None,
                x_label: str = None, y_label: str = None, legend_location: str = "lower right"):
    if not os.path.isdir(path_to_file):
        os.makedirs(path_to_file)

    fig, ax = plt.subplots()

    if main_title:
        fig.suptitle(main_title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    for y_vals in each_graph_y_vals:
        ax.plot(x_vals, y_vals[0], label=y_vals[1])

    plt.legend(loc=legend_location)
    fig.savefig(f'{path_to_file}/{file_name}', dpi=fig.dpi)
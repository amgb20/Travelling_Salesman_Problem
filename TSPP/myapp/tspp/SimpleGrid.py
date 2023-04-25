import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
from tsp_solver.greedy import solve_tsp
import time
import os
import base64
from io import BytesIO

# Function that create a timer to detect if the algorithm is taking too long


def start_clock():
    start_time = time.time()
    return start_time


def end_clock(start_time):
    elapsed_time = time.time() - start_time

    return elapsed_time


# Nearest Neighbor
def nearest_neighbour(dist_matrix, start=0):
    n = len(dist_matrix)
    path = [start]
    unvisited = set(range(n))
    unvisited.remove(start)

    while unvisited:
        current = path[-1]
        nearest_point = min(
            unvisited, key=lambda point: dist_matrix[current][point])
        path.append(nearest_point)
        unvisited.remove(nearest_point)

    return path

# 2-opt


def two_opt(dist_matrix, start=0):
    n = len(dist_matrix)
    path = nearest_neighbour(dist_matrix, start=start)
    improvement = True

    while improvement:
        improvement = False
        for i in range(n - 2):
            for j in range(i + 2, n - 1):
                if dist_matrix[path[i]][path[i + 1]] + dist_matrix[path[j]][path[j + 1]] > dist_matrix[path[i]][path[j]] + dist_matrix[path[i + 1]][path[j + 1]]:
                    path[i + 1:j + 1] = reversed(path[i + 1:j + 1])
                    improvement = True

    return path

# Christofides


def christofides(dist_matrix, start=0):
    return solve_tsp(dist_matrix, endpoints=(start, None))

# Find optimal starting positions for each algorithm


def find_optimal_start(dist_matrix, tsp_algorithm):
    min_cost = float('inf')
    optimal_start = 0

    for start in range(len(dist_matrix)):
        path = tsp_algorithm(dist_matrix, start)
        cost = sum(dist_matrix[path[i]][path[i + 1]]
                   for i in range(len(path) - 1))

        if cost < min_cost:
            min_cost = cost
            optimal_start = start

    return optimal_start


def run(length, width, tspp_algorithm):

    # generate a grid of points
    x = np.linspace(0, length - 1, length)
    y = np.linspace(0, width - 1, width)
    grid_points = np.array(list(itertools.product(x, y)))  # 2D array of points

    # compute distance matrix
    dist_matrix = squareform(pdist(grid_points))

    # Convert tsp_algorithm to a string
    algorithms = {
        'nn': nearest_neighbour,
        'two-opt': two_opt,
        'christofides': christofides
    }

    # find optimal starting position
    optimal_start = find_optimal_start(dist_matrix, algorithms[tspp_algorithm])

    # solve, compare and compute elapsed time
    start = start_clock()
    path = algorithms[tspp_algorithm](dist_matrix, optimal_start)
    elapsed_time = end_clock(start)

    # compute cost
    cost = sum(dist_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1))

    # Create a plot filename
    plot_filename = f"{tspp_algorithm}_{length}x{width}.png"

    # Plot and save the image
    plot = plot_path(path, f"{tspp_algorithm} (Optimal start: {optimal_start})", color='blue',
              filename=f"{tspp_algorithm}_path.csv", coordinates=grid_points)
    
    image_base64 = plot_to_base64_image(plot)
    
    return path, cost, elapsed_time, image_base64


# Plot paths
def plot_path(path, title, color='blue', filename=None, coordinates=None):
    plt.figure()
    plt.scatter(coordinates[:, 0], coordinates[:, 1],
                c='red', label='Waypoints')
   
    # place the robot at the optimal start
    plt.scatter(coordinates[0, 0], coordinates[0, 1],
                c='yellow', marker='s', label='Robot')
    plt.plot(coordinates[path, 0], coordinates[path, 1], c=color)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

    # Number the points according to the order of the path
    for i, point in enumerate(path):
        plt.annotate(
            str(i), (coordinates[point, 0], coordinates[point, 1]), fontsize=8, ha='right')

    if filename:
        # save the 2D coordinates of the path to a csv file
        np.savetxt(filename, coordinates, delimiter=",")
    
    return plt

def plot_to_base64_image(plot):
    buf = BytesIO()
    plot.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return image_base64

def path_coordinates_to_csv_string(path, coordinates):
    ordered_path_coordinates = coordinates[path, :]
    csv_data = "\n".join([f"{coord[0]},{coord[1]}" for coord in ordered_path_coordinates])
    return csv_data

# filename = "path.csv"

# if filename:
#     plot_path(nn_path, f"Nearest Neighbor (Optimal start: {nn_optimal_start})", color='blue', filename='nn_path.csv', coordinates=grid_points)
#     plot_path(two_opt_path, f"2-opt (Optimal start: {two_opt_optimal_start})", color='green', filename='two_opt_path.csv', coordinates=grid_points)
#     plot_path(christofides_path, f"Christofides (Optimal start: {christofides_optimal_start})", color='orange', filename='christofides_path.csv', coordinates=grid_points)
# else:
#     plot_path(nn_path, f"Nearest Neighbor (Optimal start: {nn_optimal_start})", color='blue')
#     plot_path(two_opt_path, f"2-opt (Optimal start: {two_opt_optimal_start})", color='green')
#     plot_path(christofides_path, f"Christofides (Optimal start: {christofides_optimal_start})", color='orange')

# plt.show()

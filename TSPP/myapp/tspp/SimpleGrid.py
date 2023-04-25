import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
from tsp_solver.greedy import solve_tsp
import time

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
        nearest_point = min(unvisited, key=lambda point: dist_matrix[current][point])
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
        cost = sum(dist_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1))

        if cost < min_cost:
            min_cost = cost
            optimal_start = start

    return optimal_start

def run(length, width, tspp_algorithm):

    # generate a grid of points
    x = np.linspace(0, length - 1, length)
    y = np.linspace(0, width - 1, width)
    grid_points = np.array(list(itertools.product(x, y))) # 2D array of points

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
    # path = tspp_algorithm(dist_matrix, optimal_start)
    path = algorithms[tspp_algorithm](dist_matrix, optimal_start)
    elapsed_time = end_clock(start)

    cost = sum(dist_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1))

    return path, cost, elapsed_time

# # Generate a 10x10 grid
# x = np.linspace(0, 9, 10)
# y = np.linspace(0, 9, 10)
# grid_points = np.array(list(itertools.product(x, y)))

# # Compute distance matrix
# dist_matrix = squareform(pdist(grid_points))

# # Find optimal starting positions for each algorithm
# nn_optimal_start = find_optimal_start(dist_matrix, nearest_neighbour)
# two_opt_optimal_start = find_optimal_start(dist_matrix, two_opt)
# christofides_optimal_start = find_optimal_start(dist_matrix, christofides)

# # Solve, compare and compute elapsed time for each algorithm ---------------  1 make sure it is the right way of doing it ----------------
# nn_start = start_clock()
# nn_path = nearest_neighbour(dist_matrix, nn_optimal_start)
# nn_elspased_time = end_clock(nn_start)

# two_opt_start = start_clock()
# two_opt_path = two_opt(dist_matrix, two_opt_optimal_start)
# two_opt_elapsed_time = end_clock(two_opt_start)

# christofides_time = start_clock()
# christofides_path = christofides(dist_matrix, christofides_optimal_start)
# christofides_elasped_time = end_clock(christofides_time)

# # Compute costs
# nn_cost = sum(dist_matrix[nn_path[i]][nn_path[i + 1]] for i in range(len(nn_path) - 1))
# two_opt_cost = sum(dist_matrix[two_opt_path[i]][two_opt_path[i + 1]] for i in range(len(two_opt_path) - 1))
# christofides_cost = sum(dist_matrix[christofides_path[i]][christofides_path[i + 1]] for i in range(len(christofides_path) - 1))

# # compute computational time
# nn_time = end_clock(start_clock())
# two_opt_time = end_clock(start_clock())
# christofides_time = end_clock(start_clock())

# print("Nearest Neighbor cost:", nn_cost, "and the computational time:", nn_elspased_time, "seconds")
# print("2-opt cost:", two_opt_cost, "and the computational time:", two_opt_elapsed_time, "seconds")
# print("Christofides cost:", christofides_cost,"and the computational time:", christofides_elasped_time, "seconds")

# # Plot paths
# def plot_path(path, title, color='blue', filename=None, coordinates=grid_points):
#     plt.figure()
#     plt.scatter(grid_points[:, 0], grid_points[:, 1], c='red')
#     # plt.scatter(grid_points[1:, 0], grid_points[1:, 1], c='blue', label='Waypoints')
#     plt.scatter(grid_points[0, 0], grid_points[nn_optimal_start, 1], c='yellow', marker='s', label='Robot') # place the robot at the optimal start
#     plt.plot(grid_points[path, 0], grid_points[path, 1], c=color)
#     plt.title(title)
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.legend()

#     # Number the points according to the order of the path
#     for i, point in enumerate(path):
#         plt.annotate(str(i), (grid_points[point, 0], grid_points[point, 1]), fontsize=8, ha='right')
    
#     if filename:
#         np.savetxt(filename, coordinates, delimiter=",") # save the 2D coordinates of the path to a csv file

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
import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
from tsp_solver.greedy import solve_tsp

# Nearest Neighbor
def nearest_neighbor(dist_matrix, start=0):
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
    path = nearest_neighbor(dist_matrix, start=start)
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

# Generate a 10x10 grid
x = np.linspace(0, 9, 10)
y = np.linspace(0, 9, 10)
grid_points = np.array(list(itertools.product(x, y)))

# Add robot to the middle of the grid
robot_point = np.array([4.5, 4.5])
grid_points = np.vstack([robot_point, grid_points])

# Compute distance matrix
dist_matrix = squareform(pdist(grid_points))

# Find optimal starting positions for each algorithm
nn_optimal_start = find_optimal_start(dist_matrix, nearest_neighbor)
two_opt_optimal_start = find_optimal_start(dist_matrix, two_opt)
christofides_optimal_start = find_optimal_start(dist_matrix, christofides)

# Solve and compare
nn_path = nearest_neighbor(dist_matrix, nn_optimal_start)
two_opt_path = two_opt(dist_matrix, two_opt_optimal_start)
christofides_path = christofides(dist_matrix, christofides_optimal_start)

nn_cost = sum(dist_matrix[nn_path[i]][nn_path[i + 1]] for i in range(len(nn_path) - 1))
two_opt_cost = sum(dist_matrix[two_opt_path[i]][two_opt_path[i + 1]] for i in range(len(two_opt_path) - 1))
christofides_cost = sum(dist_matrix[christofides_path[i]][christofides_path[i + 1]] for i in range(len(christofides_path) - 1))

print("Nearest Neighbor cost:", nn_cost)
print("2-opt cost:", two_opt_cost)
print("Christofides cost:", christofides_cost)

# Plot paths
def plot_path(path, title, color='blue'):
    plt.figure()
    plt.scatter(grid_points[:, 0], grid_points[:, 1], c='red')
    plt.plot(grid_points[path, 0], grid_points[path, 1], c=color)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")

    # Number the points according to the order of the path
    for i, point in enumerate(path):
        plt.annotate(str(i), (grid_points[point, 0], grid_points[point, 1]), fontsize=8, ha='right')

plot_path(nn_path, f"Nearest Neighbor (Optimal start: {nn_optimal_start})", color='blue')
plot_path(two_opt_path, f"2-opt (Optimal start: {two_opt_optimal_start})", color='green')
plot_path(christofides_path, f"Christofides (Optimal start: {christofides_optimal_start})", color='orange')

plt.show()


# import numpy as np
# import itertools
# import matplotlib.pyplot as plt
# from scipy.spatial import distance_matrix
# from scipy.spatial.distance import pdist, squareform
# from tsp_solver.greedy import solve_tsp

# # Generate a 10x10 grid
# x = np.linspace(0, 9, 10)
# y = np.linspace(0, 9, 10)
# grid_points = np.array(list(itertools.product(x, y)))

# # Add robot to the middle of the grid
# robot_point = np.array([4.5, 4.5])
# grid_points = np.vstack([robot_point, grid_points])

# # Compute distance matrix
# dist_matrix = squareform(pdist(grid_points))

# # Nearest Neighbor
# def nearest_neighbor(dist_matrix, start=0):
#     n = len(dist_matrix)
#     path = [start]
#     unvisited = set(range(n))
#     unvisited.remove(start)

#     while unvisited:
#         current = path[-1]
#         nearest_point = min(unvisited, key=lambda point: dist_matrix[current][point])
#         path.append(nearest_point)
#         unvisited.remove(nearest_point)

#     return path

# # 2-opt
# def two_opt(dist_matrix, start=0):
#     n = len(dist_matrix)
#     path = nearest_neighbor(dist_matrix, start=0) # start in the middle of the grid
#     improvement = True

#     while improvement:
#         improvement = False
#         for i in range(n - 2):
#             for j in range(i + 2, n - 1):
#                 if dist_matrix[path[i]][path[i + 1]] + dist_matrix[path[j]][path[j + 1]] > dist_matrix[path[i]][path[j]] + dist_matrix[path[i + 1]][path[j + 1]]:
#                     path[i + 1:j + 1] = reversed(path[i + 1:j + 1])
#                     improvement = True

#     return path

# # Christofides
# def christofides(dist_matrix,start=0):
#     return solve_tsp(dist_matrix, endpoints=(0, None))  # Start from the middle point (robot)

# # Add the functions to find the optimal starting position
# def cost_for_starting_position(start, algorithm):
#     if algorithm == "nearest_neighbor":
#         path = nearest_neighbor(dist_matrix, start)
#     elif algorithm == "two_opt":
#         path = two_opt(dist_matrix, start)
#     elif algorithm == "christofides":
#         path = christofides(dist_matrix, start)
#     else:
#         raise ValueError("Unknown algorithm")

#     cost = sum(dist_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1))
#     return cost

# def optimal_starting_position(algorithm):
#     n = len(grid_points)
#     costs = [cost_for_starting_position(i, algorithm) for i in range(n)]
#     optimal_start = np.argmin(costs) # optimal start computes 
#     min_cost = costs[optimal_start]
#     return optimal_start, min_cost

# # Find optimal starting positions for each algorithm
# nn_optimal_start, nn_optimal_cost = optimal_starting_position("nearest_neighbor")
# two_opt_optimal_start, two_opt_optimal_cost = optimal_starting_position("two_opt")
# christofides_optimal_start, christofides_optimal_cost = optimal_starting_position("christofides")

# print("Nearest Neighbor optimal starting position:", nn_optimal_start, "with cost:", nn_optimal_cost)
# print("2-opt optimal starting position:", two_opt_optimal_start, "with cost:", two_opt_optimal_cost)
# print("Christofides optimal starting position:", christofides_optimal_start, "with cost:", christofides_optimal_cost)

# # Solve and compare using optimal starting positions
# nn_path = nearest_neighbor(dist_matrix, nn_optimal_start)
# two_opt_path = two_opt(dist_matrix, two_opt_optimal_start)
# christofides_path = christofides(dist_matrix, christofides_optimal_start)

# # Plot paths
# def plot_path(path, title, color='blue'):
#     plt.figure()
#     plt.scatter(grid_points[:, 0], grid_points[:, 1], c='red')
#     plt.plot(grid_points[path, 0], grid_points[path, 1], c=color)
#     plt.title(title)
#     plt.xlabel("X")
#     plt.ylabel("Y")

#     # Number the points according to the order of the path
#     for i, point in enumerate(path):
#         plt.annotate(str(i), (grid_points[point, 0], grid_points[point, 1]), fontsize=8, ha='right')


# plot_path(nn_path, f"Nearest Neighbor (Optimal start: {nn_optimal_start})", color='blue')
# plot_path(two_opt_path, f"2-opt (Optimal start: {two_opt_optimal_start})", color='green')
# plot_path(christofides_path, f"Christofides (Optimal start: {christofides_optimal_start})", color='orange')


# plt.show()

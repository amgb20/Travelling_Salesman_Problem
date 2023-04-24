# add the number for each points
# create a grid
# do the tracing live
# set km as the unit
# set one agent (robot)
# set its position on the map
# create a more sexy gui with every type of method

import numpy as np
import matplotlib.pyplot as plt

# Generate random waypoints
def generate_waypoints(num_waypoints):
    return np.random.rand(num_waypoints, 2)

# Calculate distance matrix
def calculate_distance_matrix(waypoints):
    num_waypoints = len(waypoints)
    distance_matrix = np.zeros((num_waypoints, num_waypoints))
    for i in range(num_waypoints):
        for j in range(num_waypoints):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(waypoints[i] - waypoints[j])
    return distance_matrix

# Create initial population
def create_initial_population(num_individuals, num_waypoints):
    population = []
    for _ in range(num_individuals):
        individual = np.random.permutation(num_waypoints)
        population.append(individual)
    return np.array(population)

# Calculate total distance of a route
def route_distance(individual, distance_matrix):
    distance = 0
    for i in range(len(individual)):
        distance += distance_matrix[individual[i-1], individual[i]]
    return distance

# Tournament selection
def tournament_selection(population, fitness, k=2):
    selected_indices = np.random.choice(len(population), k)
    best_index = selected_indices[np.argmin(fitness[selected_indices])]
    return population[best_index]

# Ordered crossover
def ordered_crossover(parent1, parent2):
    size = len(parent1)
    start, end = np.random.choice(size, 2, replace=False)
    if start > end:
        start, end = end, start

    child1 = np.full(size, -1, dtype=int)
    child1[start:end] = parent1[start:end]

    count = 0
    for i in range(size):
        if parent2[(end + i) % size] not in child1:
            child1[(end + count) % size] = parent2[(end + i) % size]
            count += 1

    return child1

# Swap mutation
def swap_mutation(individual, mutation_rate):
    mutated_individual = np.copy(individual)
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            swap_index = np.random.randint(len(individual))
            mutated_individual[i], mutated_individual[swap_index] = mutated_individual[swap_index], mutated_individual[i]
    return mutated_individual

# Genetic algorithm
def genetic_algorithm(waypoints, num_individuals=100, num_generations=1000, mutation_rate=0.1, k=2):
    distance_matrix = calculate_distance_matrix(waypoints)
    population = create_initial_population(num_individuals, len(waypoints))
    
    best_individual = None
    best_fitness = float('inf')
    
    for _ in range(num_generations):
        fitness = np.array([route_distance(individual, distance_matrix) for individual in population])

        if fitness.min() < best_fitness:
            best_fitness = fitness.min()
            best_individual = population[np.argmin(fitness)]

        new_population = []

        for _ in range(num_individuals):
            parent1 = tournament_selection(population, fitness, k)
            parent2 = tournament_selection(population, fitness, k)

            child = ordered_crossover(parent1, parent2)
            child = swap_mutation(child, mutation_rate)

            new_population.append(child)

        population = np.array(new_population)

    return best_individual, best_fitness

# Visualize the best route
def visualize_route(waypoints, best_individual):
    ordered_waypoints = waypoints[best_individual]

    plt.figure(figsize=(10, 5))
    plt.plot(ordered_waypoints[:, 0], ordered_waypoints[:, 1], 'o-', lw=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Best route found by genetic algorithm")
    plt.show()

# Main function
def main():
    np.random.seed(42)
    
    num_waypoints = 100
    num_individuals = 100
    num_generations = 1000
    mutation_rate = 0.1
    k = 2

    waypoints = generate_waypoints(num_waypoints)
    best_individual, best_fitness = genetic_algorithm(waypoints, num_individuals, num_generations, mutation_rate, k)

    print("Best route found has a distance of:", best_fitness)
    visualize_route(waypoints, best_individual)

if __name__ == "__main__":
    main()
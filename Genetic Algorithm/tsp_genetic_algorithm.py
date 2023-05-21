import numpy as np
import matplotlib.pyplot as plt
import csv

def load_distance_matrix():
    csv_file = 'distance_matrix_TSP20.csv'
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        data = list(csv_reader)
        dist_mat = np.array(data, dtype=int)
    return dist_mat

def create_initial_population(num_individuals, num_waypoints):
    population = []
    for _ in range(num_individuals):
        individual = np.random.permutation(num_waypoints)
        population.append(individual)
    return np.array(population)

def route_distance(individual, distance_matrix):
    distance = 0
    for i in range(len(individual)):
        distance += distance_matrix[individual[i-1], individual[i]]
    return distance

def tournament_selection(population, fitness, k=2):
    selected_indices = np.random.choice(len(population), k)
    best_index = selected_indices[np.argmin(fitness[selected_indices])]
    return population[best_index]

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

def swap_mutation(individual, mutation_rate):
    mutated_individual = np.copy(individual)
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            swap_index = np.random.randint(len(individual))
            mutated_individual[i], mutated_individual[swap_index] = mutated_individual[swap_index], mutated_individual[i]
    return mutated_individual

def genetic_algorithm(distance_matrix, num_individuals=100, num_generations=1000, mutation_rate=0.1, k=2):
    num_waypoints = distance_matrix.shape[0]
    population = create_initial_population(num_individuals, num_waypoints)
    
    best_individual = None
    best_fitness = float('inf')
    avg_distances = []  # Record average distances for each generation

    for _ in range(num_generations):
        fitness = np.array([route_distance(individual, distance_matrix) for individual in population])
        avg_distances.append(np.mean(fitness))

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

        # Plotting the average distance of the population in real-time
        plt.figure(1)  # Use a single figure, you don't need to create a new one every time
        plt.clf()  # Clear the figure
        plt.plot(avg_distances)
        plt.title('Average Distance by Generation')
        plt.xlabel('Generation')
        plt.ylabel('Average Distance')
        plt.grid()
        plt.draw()  # Update the plot
        plt.pause(0.01)  # Small pause to allow the plot to update


    return best_individual, best_fitness, avg_distances

def main():
    np.random.seed(42)
    
    num_individuals = 100
    num_generations = 1000
    mutation_rate = 0.1
    k = 2

    distance_matrix = load_distance_matrix()
    best_individual, best_fitness, avg_distances = genetic_algorithm(distance_matrix, num_individuals, num_generations, mutation_rate, k)

    print("Best route found has a distance of:", best_fitness)
    plt.show()

if __name__ == "__main__":
    main()

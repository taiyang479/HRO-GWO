# -*- coding: utf-8 -*-
import numpy as np
from numpy.random import rand
from FS.functionHO import Fun

def init_position(lb, ub, N, dim):

    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()
    return X

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            s = X[i, d]
            if s > thres:
                Xbin[i, d] = 1
            else:
                Xbin[i, d] = 0

    return Xbin

def boundary(x, lb, ub):

    if x < lb:
        x = lb
    if x > ub:
        x = ub
    return x

def crossover(parent1, parent2):

    dim = len(parent1)
    cp1 = np.random.randint(1, dim - 1)
    cp2 = np.random.randint(cp1, dim)
    offspring1 = np.concatenate([parent1[:cp1], parent2[cp1:cp2], parent1[cp2:]])
    offspring2 = np.concatenate([parent2[:cp1], parent1[cp1:cp2], parent2[cp2:]])
    return offspring1, offspring2

def mutation(offspring, pm):

    for i in range(len(offspring)):
        if rand() < pm:
            offspring[i] = 1 - offspring[i]
    return offspring

def ega(xtrain, ytrain, opts):

    # Parameters
    ub = 1
    lb = 0
    thres = 0.5
    N = opts['N']
    max_iter = opts['T']
    crossover_rate = 0.7
    mutation_rate = 0.01

    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    # Initialize position
    population = init_position(lb, ub, N, dim)

    # Binary conversion
    population_bin = binary_conversion(population, thres, N, dim)

    # Fitness at first iteration
    fitness = np.zeros([N, 1], dtype='float')
    best_solution = np.zeros([1, dim], dtype='float')
    best_fitness = float('inf')

    for i in range(N):
        fitness[i, 0] = Fun(xtrain, ytrain, population_bin[i, :], opts)
        if fitness[i, 0] < best_fitness:
            best_solution[0, :] = population[i, :]
            best_fitness = fitness[i, 0]

    # Pre
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    curve[0, t] = best_fitness
    print("Iteration:", t + 1)
    print("Best (EGA):", curve[0, t])
    t += 1

    while t < max_iter:
        bestIndi = population[np.argmin(fitness[:, 0])]  # 得到当代的最优个体
        # Selection (tournament selection)
        selected_indices = np.random.choice(N, size=N-1, replace=True)
        selected_population = population[selected_indices, :]

        # Crossover
        offspring_population = []
        for i in range(0, N-1, 2):
            parent1 = selected_population[i, :]
            if i + 1 < N-1:
                parent2 = selected_population[i + 1, :]
                if rand() < crossover_rate:
                    offspring1, offspring2 = crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1, parent2
                offspring_population.append(offspring1)
                offspring_population.append(offspring2)
            else:
                offspring_population.append(parent1)

        offspring_population = np.array(offspring_population)

        # Mutation
        for i in range(N-1):
            offspring_population[i, :] = mutation(offspring_population[i, :], mutation_rate)

        # Binary conversion
        offspring_bin = binary_conversion(offspring_population, thres, N-1, dim)

        # Fitness of offspring
        offspring_fitness = np.zeros([N-1, 1], dtype='float')
        for i in range(N-1):
            offspring_fitness[i, 0] = Fun(xtrain, ytrain, offspring_bin[i, :], opts)
            if offspring_fitness[i, 0] < best_fitness:
                best_solution[0, :] = offspring_population[i, :]
                best_fitness = offspring_fitness[i, 0]

        # Update population
        population = np.vstack((bestIndi, offspring_population))
        fitness = np.vstack((np.array([[best_fitness]]), offspring_fitness))

        curve[0, t] = best_fitness
        print("Iteration:", t + 1)
        print("Best (EGA):", curve[0, t])
        t += 1


    # Best feature subset
    Gbin = binary_conversion(best_solution, thres, 1, dim)
    Gbin = Gbin.reshape(dim)
    pos = np.asarray(range(0, dim))
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)

    # Create dictionary
    ega_data = {'sf': sel_index, 'c': curve, 'nf': num_feat, 'fitness': curve[0, max_iter - 1]}

    return ega_data

import numpy as np
from numpy.random import rand
from FS.functionHO import Fun
import math
import copy


def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()
    return X

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def single_binary_conversion(X, thres, dim):
    Xbin = np.zeros([1, dim], dtype='int')
    for d in range(dim):
        s = X[d]
        if s > thres:
            Xbin[0, d] = 1
        else:
            Xbin[0, d] = 0
    return Xbin


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


def hro(xtrain, ytrain, opts):
    # Parameters
    ub = 1
    lb = 0
    thres = 0.5
    N = opts['N']
    max_iter = opts['T']

    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    # Initialize position
    X = init_position(lb, ub, N, dim)

    # Binary conversion
    Xbin = binary_conversion(X, thres, N, dim)

    # Fitness at first iteration
    fit = np.zeros([N, 1], dtype='float')

    for i in range(N):
        fit[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts)

    # Pre
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    X = np.hstack((X, fit))
    groupSize = int(N / 3)
    max_trial = 60
    X = X[np.lexsort(X.T)]
    fitness = X[:, -1]
    fitness = fitness.reshape((N, 1))
    X = X[:, 0:-1]
    best_rice = np.zeros([1, dim], dtype='float')
    best_rice[0, :] = X[0, :]
    best_fit = fitness[0]
    trial_population = np.zeros([N, 1], dtype='float')
    bin_population = np.zeros([N, dim], dtype='float')

    trial_rice = np.zeros([1, dim], dtype='float')
    sterile = np.zeros([1, dim], dtype='float')
    maintainer = np.zeros([1, dim], dtype='float')
    neighbor = np.zeros([1, dim], dtype='float')

    curve[0, t] = best_fit.copy()
    print("Iteration:", t + 1)
    print("Best (HRO):", curve[0, t])
    t += 1


    while t < max_iter:
        for k in range(2 * groupSize, N):
            m = np.random.randint(low = 0, high = groupSize)
            s = np.random.randint(low = 2 * groupSize, high = N)
            maintainer = X[m]
            sterile = X[s]

            for j in range(dim):
                flag = 1
                while flag:
                    r1 = np.random.uniform(0, 1)
                    r2 = np.random.uniform(0, 1)
                    if r1 + r2 != 0:
                        flag = 0
                trial_rice[0, j] = (r1 * maintainer[j] + r2 * sterile[j]) / (r1 + r2)
                trial_rice[0, j] = boundary(trial_rice[0, j], lb[0, j], ub[0, j])
            X_hro_bin = single_binary_conversion(trial_rice[0, :], thres, dim)
            trial_rice_fitness = Fun(xtrain, ytrain, X_hro_bin[0, :], opts)

            if trial_rice_fitness < fitness[k]:
                X[k] = trial_rice
                fitness[k] = trial_rice_fitness
                if fitness[k] < best_fit:
                    best_rice[0, :] = X[k, :]
                    best_fit = fitness[k]
                    print(k)


        for a in range(groupSize, 2 * groupSize):
            if trial_population[a] < max_trial:
                for j in range(dim):
                    n = np.random.randint(low = groupSize, high = 2 * groupSize)
                    neighbor[0, j] = X[n, j]
                    r3 = np.random.uniform(0, 1)
                    trial_rice[0, j] = r3 * (best_rice[0, j] - neighbor[0, j]) + X[a, j]
                    trial_rice[0, j] = boundary(trial_rice[0, j], lb[0, j], ub[0, j])

                X_hro_bin = single_binary_conversion(trial_rice[0, :], thres, dim)
                trial_rice_fitness = Fun(xtrain, ytrain, X_hro_bin[0, :], opts)


                if trial_rice_fitness < fitness[a]:
                    X[a] = trial_rice
                    fitness[a] = trial_rice_fitness
                    trial_population[a] = 0
                    if fitness[a] < best_fit:
                        best_rice[0, :] = X[a, :]
                        best_fit = fitness[a]
                else:
                    trial_population[a] += 1
            else:
                print("Renewal")

                for b in range(dim):
                    r4 = np.random.random()
                    X[a, b] = r4 * (ub[0, b] - lb[0, b]) + X[a, b] + lb[0, b]
                    X[a, b] = boundary(X[a, b], lb[0, b], ub[0, b])

                bin_population[a] = single_binary_conversion(X[a], thres, dim)
                fitness[a] = Fun(xtrain, ytrain, bin_population[a], opts)
                trial_population[a] = 0
                if fitness[a] < best_fit:
                    best_rice[0, :] = X[a, :]
                    best_fit = fitness[a]
                    print(a)


        trial_population = trial_population.reshape((N,1))
        X = np.hstack((X, trial_population))
        X = np.hstack((X, fitness))
        X = X[np.lexsort(X.T)]
        fitness = X[:, -1]
        fitness = fitness.reshape((N, 1))
        X = X[:, 0:-1]
        trial_population = X[:, -1]
        trial_population = trial_population.reshape((N, 1))
        X = X[:, 0:-1]

        # Binary conversion

        curve[0, t] = best_fit.copy()
        print("Iteration:", t + 1)
        print("Best (HRO):", curve[0, t])
        t += 1

    # Best feature subset
    Gbin = binary_conversion(best_rice, thres, 1, dim)
    Gbin = Gbin.reshape(dim)
    pos = np.asarray(range(0, dim))
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)

    # Create dictionary
    hro_data = {'sf': sel_index, 'c': curve, 'nf': num_feat, 'fitness': curve[0, max_iter-1] }

    return hro_data

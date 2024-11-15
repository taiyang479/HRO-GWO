from algorithm import BaseEA
from algorithm import selection
import numpy as np
import random, copy, math


class Parallel(BaseEA):
    def __init__(self, _np, n, uppers, lowers, **kwargs):
        BaseEA.__init__(self, _np, n, uppers, lowers, **kwargs)

        if self.np % 3 != 0:
            raise ValueError('in HRO, NP must be divisible by 3')

        self.group_size = int(self.np / 3)

        self.Pheromones = dict()
        self.Pheromones['0'] = np.ones(self.n)
        self.Pheromones['1'] = np.ones(self.n)

        self.Heuristic = self.importance

        #
        self.acosolutions = [self.create_solution() for _ in range(30 - 2 * self.group_size)]
        self.acobest_solution = self.create_solution()

    def fit(self, gen):
        # sort
        # maintainer restorer sterile
        self.best_solution.fitness, self.best_solution.acc = self.fitness_func(self.best_solution.binaryVector)

        # 对rice进行排序
        for rice in self.solutions:
            rice.exchange_binary_vector()
            rice.fitness, rice.acc = self.fitness_func(rice.binaryVector)
            if rice.fitness < self.best_solution.fitness:
                self.best_solution = rice
        self.solutions.sort(key=lambda s: s.fitness)

        for i in range(gen):
            self.hro_progress(gen)
            self.aco_progress(gen)


            if self.acobest_solution.fitness <= self.best_solution.fitness:
                self.solutions[self.np - 1] = copy.deepcopy(self.acobest_solution)
                self.update_pheromones(self.acobest_solution.binaryVector, self.acobest_solution.fitness, 0.3)
                self.add_best(self.acobest_solution)
            else:
                self.update_pheromones(self.best_solution.binaryVector, self.best_solution.fitness, 0.3)
                self.add_best(self.best_solution)

    def add_best(self, best):
        self.best_fitness_store.append(best.fitness)
        self.best_acc_store.append(best.acc)
        self.best_num_store.append(sum(best.binaryVector))

    def hro_progress(self, gen):
        self.hybridization_stage()
        self.selfing_stage(iter, gen)
        self.solutions.sort(key=lambda s: s.fitness)

    def aco_progress(self, gen):

        for ant in self.acosolutions:
            self.select_path(ant, alpha=1, beta=1)
            ant.examine_not_all_zero(self.uppers, self.lowers)
            ant.fitness, ant.acc = self.calculate_fitness(ant.binaryVector)
            if ant.fitness < self.acobest_solution.fitness:
                self.acobest_solution = copy.deepcopy(ant)

    def hybridization_stage(self):
        for i in range(2 * self.group_size, self.np):
            trial_solution = self.create_solution(all_zero=True)
            trial_solution.trial = self.solutions[i].trial

            for j in range(self.n):
                r1 = random.random()
                sterile_index = selection.random(2 * self.group_size, self.np, size=1, excludes=[i])
                maintainer_index = selection.random(0, self.group_size, size=1)
                trial_solution.vector[j] = r1 * self.solutions[sterile_index].vector[j] + (1 - r1) * \
                                           self.solutions[maintainer_index].vector[j]

            self.amend_component(trial_solution.vector)
            trial_solution.exchange_binary_vector()
            trial_solution.examine_not_all_zero(self.uppers, self.lowers)
            trial_solution.fitness, trial_solution.acc = self.calculate_fitness(trial_solution.binaryVector)
            lost = self.compare(trial_solution, i)

    def selfing_stage(self, iter, gen):
        for i in range(self.group_size, 2 * self.group_size):
            if not self.solutions[i].is_exceed_trial():
                trial_solution = self.create_solution(all_zero=True)
                restorer_index = selection.random(self.group_size, 2 * self.group_size, size=1, excludes=[i])
                r2 = random.random()
                trial_solution.vector = r2 * (
                        self.best_solution.vector - self.solutions[restorer_index].vector) + \
                                        self.solutions[i].vector

                self.amend_component(trial_solution.vector)
                trial_solution.exchange_binary_vector()
                trial_solution.examine_not_all_zero(self.uppers, self.lowers)
                trial_solution.fitness, trial_solution.acc = self.calculate_fitness(trial_solution.binaryVector)

                lost = self.compare(trial_solution, i)
                if lost == -1:
                    self.solutions[i].trial_increase()

            else:
                self.renewal_stage(i)

    def compare(self, trial_solution, index):
        if trial_solution.fitness < self.solutions[index].fitness:
            self.solutions[index] = trial_solution
            if trial_solution.fitness < self.best_solution.fitness:
                self.best_solution = copy.deepcopy(trial_solution)

            return 1

        return -1

    def renewal_stage(self, i):
        r3 = random.random()
        self.solutions[i].vector = r3 * (self.uppers - self.lowers) + self.solutions[i].vector + self.lowers
        self.amend_component(self.solutions[i].vector)
        self.solutions[i].exchange_binary_vector()
        self.solutions[i].examine_not_all_zero(self.uppers, self.lowers)
        self.solutions[i].fitness, self.solutions[i].acc = self.calculate_fitness(self.solutions[i].binaryVector)

    def select_path(self, ant, alpha, beta):
        for i in range(self.n):
            part1 = pow(self.Pheromones['0'][i], alpha) * pow(self.knee, beta)
            part2 = pow(self.Pheromones['1'][i], alpha) * pow(self.Heuristic[i], beta)
            p = part2 / (part1 + part2)
            rand = random.random()
            if rand < p:
                ant.binaryVector[i] = 1
                ant.vector[i] = abs(ant.vector[i])
            else:
                ant.binaryVector[i] = 0
                ant.vector[i] = -abs(ant.vector[i])

    def update_pheromones(self, vector, fitness, rho):
        temp_pheromone = fitness
        for i in range(len(vector)):
            if vector[i] == 1:
                self.Pheromones['1'][i] = (1 - rho) * self.Pheromones['1'][i] + temp_pheromone
                self.Pheromones['0'][i] = (1 - rho) * self.Pheromones['0'][i]
            else:
                self.Pheromones['0'][i] = (1 - rho) * self.Pheromones['0'][i] + temp_pheromone
                self.Pheromones['1'][i] = (1 - rho) * self.Pheromones['1'][i]

    def local_search(self):
        temp = copy.deepcopy(self.solutions[0])
        indexes = selection.random(0, self.n, size=3)
        for index in indexes:
            if temp.binaryVector[index] == 1:
                temp.binaryVector[index] = 0
            else:
                temp.binaryVector[index] = 1

        temp.fitness, temp.acc = self.calculate_fitness(temp.binaryVector)
        if temp.fitness < self.best_solution.fitness:
            self.best_solution = temp
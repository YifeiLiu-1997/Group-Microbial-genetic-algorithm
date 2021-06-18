"""
Mircobial Genetic Algorithm to find the shortest path for travel sales problem.
"""

import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas as pd

CROSS_RATE = 0.6
MUTATE_RATE = 0.05
POP_SIZE = 100
N_GENERATIONS = 200
X = np.linspace(0, N_GENERATIONS, N_GENERATIONS)
Y = []


class GA(object):
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, ):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
        self.pop = np.vstack([np.random.permutation(DNA_size) for _ in range(pop_size)])

    def translateDNA(self, DNA, city_position):                                     # get cities' coord in order
        line_x = np.empty_like(DNA, dtype=np.float64)
        line_y = np.empty_like(DNA, dtype=np.float64)
        for i, d in enumerate(DNA):
            city_coord = city_position[d]
            line_x[i, :] = city_coord[:, 0]
            line_y[i, :] = city_coord[:, 1]
        return line_x, line_y

    def get_fitness(self, line_x, line_y):
        total_distance = np.empty((line_x.shape[0],), dtype=np.float64)
        for i, (xs, ys) in enumerate(zip(line_x, line_y)):
            total_distance[i] = np.sum(np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys))))
        fitness = np.exp(self.DNA_size ** 3 / total_distance)
        return fitness, total_distance

    def crossover(self, loser, winner):
        if np.random.rand() < self.cross_rate:
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)   # choose crossover points
            keep_city = loser[~cross_points]                                        # find the city number
            swap_city = winner[np.isin(winner.ravel(), keep_city, invert=True)]
            loser[:] = np.concatenate((keep_city, swap_city))
        return [loser, winner]

    def mutate(self, loser, winner):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                swap_point = np.random.randint(0, self.DNA_size)
                swapA, swapB = loser[point], loser[swap_point]
                loser[point], loser[swap_point] = swapB, swapA
        return [loser, winner]

    def select(self, fitness):
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        return self.pop[idx]

    def evolve(self, n):
        for i in range(0, n):
            lx, ly = self.translateDNA(ga.pop, env.city_position)
            fitness, total_distance = self.get_fitness(lx, ly)
            self.pop = self.select(fitness)
            for j in range(0, int(self.pop_size/2)):
                pop_idx = [j, j*2]
                pop = self.pop[pop_idx]                                             # select two pop
                lx, ly = self.translateDNA(pop, env.city_position)
                fitness, total_distance = ga.get_fitness(lx, ly)                    # compare loser_winner fitness
                loser_idx = np.argmin(fitness)
                winner_idx = np.argmax(fitness)
                loser = pop[loser_idx]
                winner = pop[winner_idx]
                pop = self.crossover(loser, winner)                                 # crossover loser_winner
                if loser_idx == 0:
                    self.pop[pop_idx] = pop
                else:
                    self.pop[pop_idx[0]] = pop[1]
                    self.pop[pop_idx[1]] = pop[0]
                pop = self.mutate(loser, winner)                                    # mutate loser
                if loser_idx == 0:
                    self.pop[pop_idx] = pop
                else:
                    self.pop[pop_idx[0]] = pop[1]
                    self.pop[pop_idx[1]] = pop[0]

        lx, ly = ga.translateDNA(ga.pop, env.city_position)
        fitness, total_distance = ga.get_fitness(lx, ly)
        best_idx = np.argmax(fitness)
        print('Gen:', generation, '| best dict: %.2f' % total_distance[best_idx])
        Y.append(total_distance[best_idx])


class TravelSalesPerson(object):
    def __init__(self):
        data = pd.read_csv('china.csv', delimiter=';', header=None).values
        self.city_position = data[:, 1:]
        self.city_name = data[:, 0]
        plt.ion()

    def plotting(self, lx, ly, total_d):
        plt.cla()
        plt.scatter(self.city_position[:, 0].T, self.city_position[:, 1].T, s=20, c='k')
        plt.plot(lx.T, ly.T, 'r-')
        plt.text(90, 20, "Total distance=%.2f" % total_d, fontdict={'size': 16, 'color': 'black'})


plt.rcParams['font.sans-serif'] = ['SimHei']
ga = GA(DNA_size=34, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)
plt.figure()
env = TravelSalesPerson()

start_time = datetime.datetime.now()

for generation in range(N_GENERATIONS):
    ga.evolve(2)

lx, ly = ga.translateDNA(ga.pop, env.city_position)
fitness, total_distance = ga.get_fitness(lx, ly)
best_idx = np.argmax(fitness)
env.plotting(lx[best_idx], ly[best_idx], total_distance[best_idx])
plt.figure()
plt.plot(X, Y)
plt.xlabel('迭代次数')
end_time = datetime.datetime.now()
print('N_CITIES:', 34, '\nCROSS_RATE:' , CROSS_RATE, '\nMUTATE_RATE:', MUTATE_RATE,
      '\nPOP_SIZE:', POP_SIZE, '\nN_GENERATIONS:', N_GENERATIONS, '\nTIME: ', end_time-start_time)


plt.ioff()
plt.show()

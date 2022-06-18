import numpy
from multiprocessing import Pool
from ga.individ import Individual


class Population:
    def __init__(self, population_size, crossover_count, track, start_coord, seed=None, order_changed=False):
        self.individuals = []
        self.crossover_count = crossover_count
        self.genes_count = 3
        self.track = track
        self.start_coord = start_coord
        for i in range(population_size):
            self.individuals.append(Individual(order_changed))
        if seed:
            for i in range(min(len(seed), population_size)):
                ind = Individual(order_changed)
                ind.genes = numpy.array(seed[i])
                self.individuals.append(ind)

    def calculate_fitness(self):
        with Pool(4) as pool:
            results = [pool.apply_async(ind.calculate_fitness,
                                        (self.track.copy(),
                                         self.start_coord,
                                         True
                                         )
                                        )
                       for ind in self.individuals]
            for i, result in enumerate(results):
                self.individuals[i] = result.get()

    def get_best_ids(self):
        best_ids = [self.individuals.index(x) for x in
                    sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)[:self.crossover_count]]
        return best_ids

    def get_worst_id(self):
        worst = min(self.individuals, key=lambda ind: ind.fitness)
        return self.individuals.index(worst)

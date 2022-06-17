import numpy
import cv2

import random
from multiprocessing import Pool

from ga.population import Population
from ga.individ import Individual


class GA_Runner:
    def __init__(self, population_count: int, track, start_coord, seed=None):
        self.population = Population(population_count, 5, track, start_coord, seed)
        self.best = []
        self.max_fitness = None  # TODO
        self.track = track
        self.start_coord = start_coord
        self.to_show = False

    def run(self):
        converge = False
        counter = 0
        generation = 1
        conv_value = None
        best_fitness = -1
        while not converge:
            self.add_fittest()
            best_fitness = self.population.individuals[
                self.population.get_best_ids()[0]
            ].fitness

            if conv_value:
                if conv_value == best_fitness:
                    counter += 1
                else:
                    print("Generation: " + str(generation) +
                          " -> Fittest = " +
                          str(best_fitness) + "\nPID = " + str(self.population.individuals[
                                                                   self.population.get_best_ids()[0]
                                                               ]))
                    with open("results.txt", "w+") as f:
                        f.write("Generation: " + str(generation) +
                                " -> Fittest = " +
                                str(best_fitness) + "\nPID = " + str(self.population.individuals[
                                                                         self.population.get_best_ids()[0]
                                                                     ]))
                    cur_best = self.population.individuals[
                        self.population.get_best_ids()[0]
                    ]
                    cur_best.did_fitness = False
                    cur_best.calculate_fitness(self.track, self.start_coord, to_show=True)
                    conv_value = best_fitness
            else:
                print("Generation: " + str(generation) +
                      " -> Fittest = " +
                      str(best_fitness) + "\nPID = " + str(self.population.individuals[
                                                               self.population.get_best_ids()[0]
                                                           ]))
                with open("results.txt", "w+") as f:
                    f.write("Generation: " + str(generation) +
                            " -> Fittest = " +
                            str(best_fitness) + "\nPID = " + str(self.population.individuals[
                                                                     self.population.get_best_ids()[0]
                                                                 ]))
                cur_best = self.population.individuals[
                    self.population.get_best_ids()[0]
                ]
                cur_best.did_fitness = False
                cur_best.calculate_fitness(self.track, self.start_coord, to_show=True)
                conv_value = best_fitness
                counter = 1

            self.selection()

            self.crossover()

            if (numpy.random.randint(10) + 1) % 3 == 0:
                self.mutation()

            if counter >= 10:
                converge = self.is_converged()
                counter = 1
            generation += 1
        print("Generation: " + str(generation) +
              " -> Fittest = " +
              str(best_fitness))
        best_ids = self.population.get_best_ids()
        self.population.individuals[best_ids[0]].print()

    def selection(self):
        best_ids = self.population.get_best_ids()
        self.best = (numpy.array(self.population.individuals)[numpy.array(best_ids)]).tolist()

    def crossover(self):
        new_best = []
        for ind1_i in range(len(self.best)):
            for ind2_i in range(ind1_i + 1, len(self.best)):
                crossover_index = numpy.random.randint(self.population.genes_count)
                ind_1 = Individual()
                ind_1.genes = self.best[ind1_i].genes.copy()
                ind_2 = Individual()
                ind_2.genes = self.best[ind2_i].genes.copy()
                ind_3 = self.get_average_ind(ind_1, ind_2)
                self._swap_genes(ind_1, ind_2, crossover_index)
                ind_1.reset()
                ind_2.reset()
                ind_3.reset()
                new_best.append(ind_1)
                new_best.append(ind_2)
                new_best.append(ind_3)
        self.best = new_best

    def mutation(self):
        new_best = self.best.copy()
        for ind in new_best:
            mut_id = numpy.random.randint(self.population.genes_count - 1)
            t = ind.genes
            t[mut_id] = round(random.uniform(-10, 10), 3)
            ind.genes = t
            ind.reset()
        self.best.extend(new_best)

    def add_fittest(self):
        self.best.extend(self.population.individuals)
        with Pool(4) as pool:
            results = [pool.apply_async(ind.calculate_fitness,
                                        (self.track.copy(),
                                         self.start_coord,
                                         self.to_show)
                                        )
                       for ind in self.best]
            for i, result in enumerate(results):
                self.best[i] = result.get()

        self.best.sort(key=lambda ind: ind.fitness, reverse=True)
        self.population.individuals = self.best[:len(self.population.individuals)]


    @staticmethod
    def _swap_genes(first, second, crossover_point):
        first_genes = first.genes
        second_genes = second.genes
        for i in range(crossover_point + 1):
            t = first_genes[i]
            first_genes[i] = second_genes[i]
            second_genes[i] = t
        first.genes = first_genes
        second.genes = second_genes

    def is_converged(self) -> bool:
        counter = {}
        for ind1 in self.population.individuals:
            for ind2 in self.population.individuals:
                if all((ind2.genes - ind2.genes) <= 0.0001):
                    if str(ind1.genes) in counter.keys():
                        counter[str(ind1.genes)] = counter[str(ind1.genes)] + 1
                    else:
                        counter[str(ind1.genes)] = 1
        for value in counter.values():
            if value >= len(self.population.individuals) * 0.7:
                return True
        return False

    def get_average_ind(self, ind_1: Individual, ind_2: Individual):
        new_ind = Individual()
        new_ind.genes = numpy.array([(ind_1.genes[0] + ind_2.genes[0]) / 2,
                                     (ind_1.genes[1] + ind_2.genes[1]) / 2,
                                     (ind_1.genes[2] + ind_2.genes[2]) / 2])
        return new_ind

    def set_show(self):
        print("set_show")
        self.to_show = not self.to_show

from functools import total_ordering
import random
import cv2
import numpy
from simulation.PID_controller import PID
from simulation.car import Car
from simulation.pixy import Pixy
import gc


@total_ordering
class Fitness:
    def __init__(self, order_changed=False):
        self.value1 = 0  # max section reached
        self.value2 = 1  # steps taken
        self.value3 = 0  # sum coverage
        self.order = self._order1
        if order_changed:
            self.order = self._order2

    def __eq__(self, other):
        if other is None or not isinstance(other, Fitness):
            raise Exception("Invalid cmp with object" + str(other))
        return self.value1 == other.value1 and self.value2 == other.value2 and abs(self.value3 - other.value3) < 0.0001

    def __lt__(self, other):
        return self.order(other)

    def _order1(self, other):
        if other is None or not isinstance(other, Fitness):
            raise Exception("Invalid cmp with object" + str(other))
        if self.value1 == other.value1:
            if self.value2 == other.value2:
                return (self.value3 / self.value2) < (other.value3 / other.value2)
            return self.value2 > other.value2
        return self.value1 < other.value1

    def _order2(self, other):
        if other is None or not isinstance(other, Fitness):
            raise Exception("Invalid cmp with object" + str(other))
        if self.value1 == other.value1:
            if (self.value3 / self.value2) == (other.value3 / other.value2):
                return self.value2 > other.value2
            return (self.value3 / self.value2) < (other.value3 / other.value2)
        return self.value1 < other.value1

    def __str__(self):
        return "Value1 = " + str(self.value1) + "\nValue2 = " + str(self.value2) + "\nValue3 = " + str(self.value3)

    def reset(self):
        self.value1 = 0  # max section reached
        self.value2 = 1  # steps taken
        self.value3 = 0  # sum coverage


class Individual:
    g_id = 0

    def __init__(self, order_changed=False):
        self.genes = numpy.array(
            [round(random.uniform(-10, 10), 3), round(random.uniform(-10, 10), 3), round(random.uniform(-10, 10), 3)])
        self.fitness = Fitness(order_changed)
        self.m_id = Individual.g_id
        Individual.g_id += 1
        self.did_fitness = False

    def calculate_fitness(self, track, start_coord, to_show=False, to_record=False, file_name=None):
        gc.collect()
        if self.did_fitness:
            return self
        self.fitness.reset()
        pid = PID(self.genes[0], self.genes[1], self.genes[2])
        car = Car(track, start_coord, to_record=to_record, file_name=file_name)
        pixy = Pixy(car)
        blind_counter = 0
        locations = {}
        while blind_counter <= 20:
            if to_show:
                car.show(name=str(self.m_id))
                cv2.waitKey(1)

            ((x1, y1), (x2, y2)) = pixy.get_direcetion()

            if x1 == -1 and y1 == -1:  # camera can't see path
                blind_counter += 1
                if to_show and blind_counter > 20:
                    print("\nBlindness value\n")
            else:
                blind_counter = 0

            error = numpy.rad2deg(numpy.arctan2(y2 - (car.ch - 1) / 2, (car.cw - 1) - x2))
            if error == 0:
                pid.reset()
            pid.update(error)
            alpha = pid.correction
            # print(error)
            # print(alpha)
            # print("x2 = " + str(x2) + "\ny2 = " + str(y2) + "\n")
            if alpha > 75:
                car.move(200, -200)
            elif 75 >= alpha > 60:
                car.move(200, -100)
            elif 60 >= alpha > 45:
                car.move(200, 0)
            elif 45 >= alpha > 30:
                car.move(100, 0)
            elif 30 >= alpha > 15:
                car.move(200, 100)
            elif 15 >= alpha > 0:
                car.move(200, 200)
            elif alpha == 0:
                car.move(200, 200)
            elif -15 <= alpha < 0:
                car.move(200, 200)
            elif -30 <= alpha < -15:
                car.move(100, 200)
            elif -45 <= alpha < -30:
                car.move(0, 100)
            elif -60 <= alpha < -45:
                car.move(0, 200)
            elif -75 <= alpha < -60:
                car.move(-100, 200)
            elif alpha < -75:
                car.move(-200, 200)
            new_value1 = int(numpy.max(car.get_car_center()))
            if new_value1 > 0 and self.fitness.value1 - new_value1 > 16:
                if to_show:
                    print("\nFitness value\n")
                break
            elif self.fitness.value1 < new_value1:
                self.fitness.value1 = max(new_value1, self.fitness.value1)

            self.fitness.value2 += 1
            self.fitness.value3 += car.get_line_coverage()
            if (car.x, car.y) in locations.keys():
                locations[(car.x, car.y)] += 1
            else:
                locations[(car.x, car.y)] = 1
            if max(locations.values()) >= 30:
                if to_show:
                    print("\nLocation value\n")
                break
        # cv2.waitKey()
        if to_show:
            cv2.destroyWindow(str(self.m_id))
        self.did_fitness = True
        return self

    def reset(self):
        self.did_fitness = False
        self.fitness.reset()

    def print(self):
        print("P = " + str(self.genes[0]) + "\nI = " + str(self.genes[1]) + "\nD = " + str(self.genes[2]))

    def __str__(self):
        return "P = " + str(self.genes[0]) + "\nI = " + str(self.genes[1]) + "\nD = " + str(self.genes[2])

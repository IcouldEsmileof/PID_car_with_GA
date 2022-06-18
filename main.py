import cv2
import numpy
import numpy as np
import cv2 as cv
import keyboard
from multiprocessing import freeze_support
from simulation.car import Car
from simulation.pixy import Pixy
from ga.individ import Individual
from ga.runner import GA_Runner


def exp1(img):
    ga = GA_Runner(10, img, [140, 650, 0])
    ga.run()
    return True


def exp2(img):
    ga = GA_Runner(20, img, [140, 650, 0])
    ga.run()
    return True


def exp3(img):
    ga = GA_Runner(20, img, [140, 650, 0],
                   [
                       [2.924, 0.626, 2.684],
                       [6.098, 0.964, -3.263],
                   ]
                   )
    ga.run()
    return True


def exp4(img):
    ga = GA_Runner(10, img, [140, 650, 0],
                   [
                       [2.117, 0.598, 2.684],
                       [2.924, 0.626, 2.684],
                       [6.098, 0.964, -3.263],
                       [6.098, -0.055, -3.263],
                       [4.374, -0.055, -3.263],
                       [4.097, -0.055, -3.263],
                   ],
                   order_changed=True
                   )
    ga.run()
    return True


def exp5(img):
    ga = GA_Runner(10, img, [140, 650, 0], order_changed=True)
    ga.run()
    return True


def show_best_of_exp1(img):
    ind = Individual()
    ind.genes = numpy.array([2.298, 0.023000000000000576, -0.662])
    ind.calculate_fitness(track=img, start_coord=[140, 650, 0], to_show=True)
    print(str(ind))
    print("Criteria 1: " + str(ind.fitness.value1))
    print("Criteria 2: " + str(ind.fitness.value2))
    print("Criteria 3: " + str(ind.fitness.value3 / ind.fitness.value2))
    return True


def show_best_of_exp2(img):
    ind = Individual()
    ind.genes = numpy.array([6.098, 0.964, -3.263])
    ind.calculate_fitness(track=img, start_coord=[140, 650, 0], to_show=True)
    print(str(ind))
    print("Criteria 1: " + str(ind.fitness.value1))
    print("Criteria 2: " + str(ind.fitness.value2))
    print("Criteria 3: " + str(ind.fitness.value3 / ind.fitness.value2))
    return True


def show_best_of_exp3(img):
    ind = Individual()
    ind.genes = numpy.array([4.097, -0.055, -3.263])
    ind.calculate_fitness(track=img, start_coord=[140, 650, 0], to_show=True)
    print(str(ind))
    print("Criteria 1: " + str(ind.fitness.value1))
    print("Criteria 2: " + str(ind.fitness.value2))
    print("Criteria 3: " + str(ind.fitness.value3 / ind.fitness.value2))
    return True


def show_best_of_exp4(img):
    ind = Individual(order_changed=True)
    ind.genes = numpy.array([1.016, 0.016, 1.19725])
    ind.calculate_fitness(track=img, start_coord=[140, 650, 0], to_show=True)
    print(str(ind))
    print("Criteria 1: " + str(ind.fitness.value1))
    print("Criteria 2: " + str(ind.fitness.value3 / ind.fitness.value2))
    print("Criteria 3: " + str(ind.fitness.value2))
    return True


def show_best_of_exp5(img):
    ind = Individual(order_changed=True)
    ind.genes = numpy.array([9.485, 9.869, -7.637])
    ind.calculate_fitness(track=img, start_coord=[140, 650, 0], to_show=True)
    print(str(ind))
    print("Criteria 1: " + str(ind.fitness.value1))
    print("Criteria 2: " + str(ind.fitness.value3 / ind.fitness.value2))
    print("Criteria 3: " + str(ind.fitness.value2))
    return True


def main():
    img = cv.imread("ga_track.png")[:, :, 0]

    A = {"x": 240, "y": 715}
    B = {"x": 240, "y": 300}
    C = {"x": 240, "y": 460}
    D = {"x": 540, "y": 460}
    E = {"x": 540, "y": 300}
    F = {"x": 835, "y": 460}
    G = {"x": 835, "y": 265}
    H = {"x": 835, "y": 300}
    I = {"x": 865, "y": 650}
    J = {"x": 865, "y": 715}
    K = {"x": 1100, "y": 650}
    L = {"x": 1100, "y": 825}
    M = {"x": 865, "y": 825}
    N = {"x": 735, "y": 700}
    O = {"x": 550, "y": 715}
    P = {"x": 865, "y": 460}

    img[B["y"]:A["y"], 0:A["x"]][img[B["y"]:A["y"], 0:A["x"]] > 0] = 1 * 16 - 1  # section 1
    img[0:B["y"], 0:B["x"]][img[0:B["y"], 0:B["x"]] > 0] = 2 * 16 - 1  # section 2
    img[0:B["y"], B["x"]:E["x"]][img[0:B["y"], B["x"]:E["x"]] > 0] = 3 * 16 - 1  # section 3
    img[B["y"]:C["y"], B["x"]:E["x"]][img[B["y"]:C["y"], B["x"]:E["x"]] > 0] = 4 * 16 - 1  # section 4
    img[E["y"]:D["y"], E["x"]:H["x"]][img[E["y"]:D["y"], E["x"]:H["x"]] > 0] = 5 * 16 - 1  # section 5
    img[0:E["y"], E["x"]:H["x"]][img[0:E["y"], E["x"]:H["x"]] > 0] = 6 * 16 - 1  # section 6
    img[0:G["y"], G["x"]:][img[0:G["y"], G["x"]:] > 0] = 7 * 16 - 1  # section 7
    img[G["y"]:F["y"], G["x"]:][img[G["y"]:F["y"], G["x"]:] > 0] = 8 * 16 - 1  # section 8
    img[P["y"]:I["y"], P["x"]:][img[P["y"]:I["y"], P["x"]:] > 0] = 9 * 16 - 1  # section 9
    img[I["y"]:M["y"], I["x"]:K["x"]][img[I["y"]:M["y"], I["x"]:K["x"]] > 0] = 10 * 16 - 1  # section 10
    img[K["y"]:L["y"], K["x"]:][img[K["y"]:L["y"], K["x"]:] > 0] = 11 * 16 - 1  # section 11
    img[M["y"]:, M["x"]:][img[M["y"]:, M["x"]:] > 0] = 12 * 16 - 1  # section 12
    img[J["y"]:, N["x"]:J["x"]][img[J["y"]:, N["x"]:J["x"]] > 0] = 13 * 16 - 1  # section 13
    img[C["y"]:A["y"], O["x"]:J["x"]][img[C["y"]:A["y"], O["x"]:J["x"]] > 0] = 14 * 16 - 1  # section 14
    img[N["y"]:, O["x"]:N["x"]][img[N["y"]:, O["x"]:N["x"]] > 0] = 15 * 16 - 1  # section 15
    img[O["y"]:, 0:O["x"]][img[O["y"]:, 0:O["x"]] > 0] = 16 * 16 - 1  # section 16

    l = False
    # l = exp1(img)
    # l = exp2(img)
    # l = exp3(img)
    # l = exp4(img)
    # l = exp5(img)
    # l = show_best_of_exp1(img)
    # l = show_best_of_exp2(img)
    # l = show_best_of_exp3(img)
    # l = show_best_of_exp4(img)
    # l = show_best_of_exp5(img)
    if not l:
        print(
            "My developer is lazy so he didn't make me a menu. \n"
            "Please open main.py and uncomment the line you want to run (remove the '#' symbol).\n"
            "Thank you!"
        )


if __name__ == "__main__":
    freeze_support()
    main()

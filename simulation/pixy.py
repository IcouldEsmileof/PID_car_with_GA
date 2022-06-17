import cv2 as cv
import numpy


class Pixy:
    def __init__(self, m_car):
        self.car = m_car

    def get_direcetion(self):
        # image = np.array(n,m)
        img = self.car.get_image()
        (n, m) = img.shape
        # magic
        img[img > 0] = 255
        # cv.imshow("pixy", img)
        # cv.waitKey(1)
        tl = 0
        tr = m - 1
        bl = 0
        br = m - 1
        c = numpy.array([False, False, False, False])
        while len(c[c == False]) != 0 and tl < m and tr >= 0 and bl < m and br >= 0:
            if img[0, tl] != 0:
                c[0] = True
            else:
                tl += 1
            if img[0, tr] != 0:
                c[1] = True
            else:
                tr -= 1
            if img[n - 1, bl] != 0:
                c[2] = True
            else:
                bl += 1
            if img[n - 1, br] != 0:
                c[3] = True
            else:
                br -= 1

        lt = 0
        lb = n - 1
        rt = 0
        rb = n - 1
        c = numpy.array([False, False, False, False])

        while len(c[c == False]) != 0 and lt < n and lb >= 0 and rt < n and rb >= 0:
            if img[lt, 0] != 0:
                c[0] = True
            else:
                lt += 1
            if img[lb, 0] != 0:
                c[1] = True
            else:
                lb -= 1
            if img[rt, m - 1] != 0:
                c[2] = True
            else:
                rt += 1
            if img[rb, m - 1] != 0:
                c[3] = True
            else:
                rb -= 1

        if tl <= tr and bl <= br:  # line |
            return (n - 1, (bl + br) / 2), (0, (tl + tr) / 2)
        elif lt <= lb and rt <= lb:  # line -
            result = (
                ((lt + lb) / 2, 0),
                ((rt + rb) / 2, m - 1)
            )
            if (lt + lb) / 2 < (rt + rb) / 2:
                result = (
                    ((rt + rb) / 2, m - 1),
                    ((lt + lb) / 2, 0)
                )
            return result
        else:
            if bl <= br:
                if lt <= lb:
                    return (n - 1, (bl + br) / 2), ((lt + lb) / 2, 0)
                elif rt <= rb:
                    return (n - 1, (bl + br) / 2), ((rt + rb) / 2, m - 1)
                else:
                    return (-1, -1), (0, m / 2)
            elif tl <= tr:
                if lt <= lb:
                    return ((lt + lb) / 2, 0), (0, (tl + tr) / 2)
                elif rt <= rb:
                    return ((rt + rb) / 2, m - 1), (0, (tl + tr) / 2)
                else:
                    return (-1, -1), (0, m / 2)
            else:
                return (-1, -1), (0, m / 2)

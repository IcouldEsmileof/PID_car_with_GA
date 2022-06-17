import cv2 as cv
import numpy


class Car:
    _me = None

    def __init__(self, track: numpy.array, start_coord: numpy.array, to_record=False, file_name=None):
        Car._me = self
        self.track = track
        self.x = start_coord[0]  # car center x
        self.y = start_coord[1]  # car center y
        self.rotation = start_coord[2]  # car rotation
        self.w = 40  # car width
        self.h = 50  # car height
        self.cx = self.x + (self.h // 2) * numpy.cos(numpy.deg2rad(self.rotation - 90))  # camera center x
        self.cy = self.y + (self.h // 2) * numpy.sin(numpy.deg2rad(self.rotation - 90))  # camera center y
        self.cw = 20  # camera width
        self.ch = 20  # camera height
        self.ccw = 10  # checked area width
        self.cch = 10  # checked area height
        self.to_show = True
        self.recorder = None
        _file_name = file_name if file_name else "output.avi"
        if to_record:
            fourcc = cv.VideoWriter_fourcc(*'XVID')
            self.recorder = cv.VideoWriter(_file_name, fourcc, 20.0, (self.track.shape[1], self.track.shape[0]))

    def __del__(self):
        if self.recorder:
            self.recorder.release()

    def get_image(self) -> numpy.array:
        # result = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0],
        #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255]]
        # return numpy.array(result, dtype=numpy.uint8)
        self.cx = self.x + (self.h // 2) * numpy.cos(numpy.deg2rad(self.rotation - 90))
        self.cy = self.y + (self.h // 2) * numpy.sin(numpy.deg2rad(self.rotation - 90))

        rot_M = cv.getRotationMatrix2D((self.x, self.y), self.rotation, 1)
        rotated_track = numpy.zeros(shape=(self.track.shape[0], self.track.shape[1], 3), dtype=numpy.uint8)
        rotated_track[:, :, 0] = self.track
        rotated_track[:, :, 2] = numpy.bitwise_or(self.draw_camera(), self.draw_car())
        rotated_track = cv.warpAffine(rotated_track, rot_M, (self.track.shape[1], self.track.shape[0]))

        cx = int(self.x + (self.h // 2) * numpy.cos(numpy.deg2rad(-90)))
        cy = int(self.y + (self.h // 2) * numpy.sin(numpy.deg2rad(-90)))
        rotated_track[cy - 10: cy + 10, cx - 10:cx + 10, 1] = 255

        # cv.imshow("rot_track", rotated_track)
        # cv.waitKey(1)
        result = rotated_track[cy - 10: cy + 10, cx - 10:cx + 10, 0]
        return result

    def show(self, name="ing"):
        img = numpy.zeros(shape=(self.track.shape[0], self.track.shape[1], 3), dtype=numpy.uint8)

        img[:, :, 0] = self.track
        img[:, :, 1] = self.draw_car()
        img[:, :, 2] = self.draw_camera()
        img[int(self.y - 2):int(self.y + 2), int(self.x - 2):int(self.x + 2), :] = 255
        img[img > 0] = 255
        if self.recorder:
            self.recorder.write(img)
        cv.imshow(name, img)

    def draw_car(self):
        car_img = numpy.zeros(shape=(self.track.shape[0], self.track.shape[1], 1), dtype=numpy.uint8)
        rect = ((int(self.x), int(self.y)), (self.w, self.h), self.rotation)
        box = cv.boxPoints(rect)
        box = numpy.int0(box)
        cv.drawContours(car_img, [box], 0, (255, 255, 255), -1)
        return car_img[:, :, 0]

    def draw_camera(self):
        camera_img = numpy.zeros(shape=(self.track.shape[0], self.track.shape[1], 1), dtype=numpy.uint8)
        self.cx = self.x + (self.h // 2) * numpy.cos(numpy.deg2rad(self.rotation - 90))
        self.cy = self.y + (self.h // 2) * numpy.sin(numpy.deg2rad(self.rotation - 90))

        rect = ((int(self.cx), int(self.cy)), (self.cw, self.ch), self.rotation)
        box = cv.boxPoints(rect)
        box = numpy.int0(box)
        cv.drawContours(camera_img, [box], 0, (255, 255, 255), -1)
        return camera_img[:, :, 0]

    def get_car_center(self):
        cc_img = numpy.zeros(shape=(self.track.shape[0], self.track.shape[1], 1), dtype=numpy.uint8)
        rect = ((int(self.x), int(self.y)), (self.ccw, self.cch), self.rotation)
        box = cv.boxPoints(rect)
        box = numpy.int0(box)
        cv.drawContours(cc_img, [box], 0, (255, 255, 255), -1)
        cc_img = numpy.bitwise_and(self.track, cc_img[:, :, 0])
        return cc_img[int(self.y) - 5: int(self.y) + 5, int(self.x) - 5:int(self.x) + 5]

    def get_line_coverage(self):
        cc_img = self.get_car_center()
        result = numpy.count_nonzero(cc_img)
        return result / (self.ccw * self.cch)

    def move(self, left, right):
        self.rotation += (left - right) // 100
        self.rotation = self.rotation if self.rotation < 360 else self.rotation - 360
        if left >= 0 and right >= 0:
            self.x = self.x + (((left + right) / 2) / 100) * numpy.cos(numpy.deg2rad(self.rotation - 90))
            self.y = self.y + (((left + right) / 2) / 100) * numpy.sin(numpy.deg2rad(self.rotation - 90))
        elif left <= 0 and right <= 0:
            self.x = self.x + (((left + right) / 2) / 100) * numpy.cos(numpy.deg2rad(self.rotation - 90))
            self.y = self.y + (((left + right) / 2) / 100) * numpy.sin(numpy.deg2rad(self.rotation - 90))

    def is_on_track(self):
        return self.get_line_coverage() > 0.2

    @classmethod
    def forward(cls):
        Car._me.x = Car._me.x + numpy.cos(numpy.deg2rad(Car._me.rotation - 90))
        Car._me.y = Car._me.y + numpy.sin(numpy.deg2rad(Car._me.rotation - 90))

    @classmethod
    def backward(cls):
        Car._me.x = Car._me.x - numpy.cos(numpy.deg2rad(Car._me.rotation - 90))
        Car._me.y = Car._me.y - numpy.sin(numpy.deg2rad(Car._me.rotation - 90))

    @classmethod
    def left(cls):
        Car._me.rotation = Car._me.rotation - 10

    @classmethod
    def right(cls):
        Car._me.rotation = Car._me.rotation + 10

    @classmethod
    def stop(cls):
        Car._me.to_show = False

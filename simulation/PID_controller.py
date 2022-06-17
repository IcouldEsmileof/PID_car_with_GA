
class PID:
    PID_MAX_INTEGRAL = 2000

    def __init__(self, p, i, d):
        self.prevError = None
        self.correction = 0.0
        self.integral = 0.0
        self.p = p
        self.i = i
        self.d = d

    def update(self, error: float):
        if self.prevError is not None:
            self.integral += error
            if self.integral > self.PID_MAX_INTEGRAL:
                self.integral = self.PID_MAX_INTEGRAL
            elif self.integral < -self.PID_MAX_INTEGRAL:
                self.integral = -self.PID_MAX_INTEGRAL
            self.correction = \
                error * self.p + \
                self.integral * self.i + \
                (error - self.prevError) * self.d
        self.prevError = error

    def reset(self):
        self.correction = 0.0
        self.integral = 0.0
        self.prevError = None

    def __str__(self):
        return "P="+str(self.p)+"\nI="+str(self.i)+"\nD="+str(self.d)
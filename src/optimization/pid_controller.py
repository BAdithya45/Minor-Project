class PIDController:
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.previous_error = 0.0

    def compute(self, setpoint, measured_value):
        error = setpoint - measured_value
        self.integral += error
        derivative = error - self.previous_error
        self.previous_error = error

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return output

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0

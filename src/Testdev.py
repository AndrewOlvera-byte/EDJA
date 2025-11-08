from RpiMotorLib import RpiMotorLib
from time import sleep

motorPins = [17,18,27,22]

# simple half-step sequence
myMotor = RpiMotorLib.BYJMotor("28BYJ48",motor_pins = None)

myMotor.motor_run(motorPins, 0.002, 4096, False, False, "half", 0.05)


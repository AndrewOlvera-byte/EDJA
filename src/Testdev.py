from RpiMotorLib import RpiMotorLib
from time import sleep
import RPi.GPIO as GPIO

# GPIO pins connected to ULN2003 driver IN1-IN4
motorPins = [17, 18, 27, 22]


# Initialize motor
myMotor = RpiMotorLib.BYJMotor("MyMotor", "28BYJ")

# Rotate 1 full revolution (~4096 half-steps)
# motor_run(pins, stepdelay, steps, clockwise, verbose, steptype, initdelay)
myMotor.motor_run(
    motorPins,    # GPIO pins
    0.002,        # step delay
    4096,         # number of steps (1 rev for 28BYJ-48)
    False,        # clockwise or counterclockwise
    False,         # verbose output
    "half",       # step type: 'half' or 'full'
    0.05          # initial delay
)

GPIO.cleanup()

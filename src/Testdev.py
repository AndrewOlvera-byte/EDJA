from RpiMotorLib import RpiMotorLib
from time import sleep

# GPIO pins connected to ULN2003 driver IN1-IN4
motorPins = [17, 18, 27, 22]

coil1 = OutputDevice(17)
coil2 = OutputDevice(18)
coil3 = OutputDevice(27)
coil4 = OutputDevice(22)

# Initialize motor
myMotor = RpiMotorLib.BYJMotor("MyMotor", motor_type="28BYJ")

# Rotate 1 full revolution (~4096 half-steps)
# motor_run(pins, stepdelay, steps, clockwise, verbose, steptype, initdelay)
myMotor.motor_run(
    motorPins,    # GPIO pins
    0.002,        # step delay
    4096,         # number of steps (1 rev for 28BYJ-48)
    False,        # clockwise or counterclockwise
    True,         # verbose output
    "half",       # step type: 'half' or 'full'
    0.05          # initial delay
)

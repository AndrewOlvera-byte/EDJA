from gpiozero import OutputDevice
from time import sleep

coil1 = OutputDevice(17)
coil2 = OutputDevice(18)
coil3 = OutputDevice(27)
coil4 = OutputDevice(22)

# simple half-step sequence
halfstep_seq = [
    [1,0,0,0],
    [1,1,0,0],
    [0,1,0,0],
    [0,1,1,0],
    [0,0,1,0],
    [0,0,1,1],
    [0,0,0,1],
    [1,0,0,1]
]

while True:
    for step in halfstep_seq:
        coil1.value, coil2.value, coil3.value, coil4.value = step
        sleep(0.001)

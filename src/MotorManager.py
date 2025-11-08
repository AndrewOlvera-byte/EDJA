import RPi.GPIO as GPIO
import time

IN1 = 17
IN2 = 18
IN3 = 27
IN4 = 22

GPIO.setmode(GPIO.BCM)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)

halfstep_seq = [
    [1,0,0,1],
    [1,0,0,0],
    [1,1,0,0],
    [0,1,0,0],
    [0,1,1,0],
    [0,0,1,0],
    [0,0,1,1],
    [0,0,0,1]
]

def move_steps(steps, delay = 0.001):
    for _ in range(steps):
        for halfstep in range(8):
            GPIO.output(IN1, halfstep_seq[halfstep][0])
            GPIO.output(IN2, halfstep_seq[halfstep][1])
            GPIO.output(IN3, halfstep_seq[halfstep][2])
            GPIO.output(IN4, halfstep_seq[halfstep][3])
            time.sleep(delay)
try:
    print("Rotating clockwise")
    move_steps(512)
    time.sleep(1)

finally:
    GPIO.cleanup()

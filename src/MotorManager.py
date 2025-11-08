import RPi.GPIO as GPIO
import time
import threading
class MotorManager:

    def __init__(self, gpio_pins):
        self.pins = gpio_pins

        # setting up
        GPIO.setmode( GPIO.BCM )
        GPIO.setup( self.pins[0], GPIO.OUT )
        GPIO.setup( self.pins[1], GPIO.OUT )
        GPIO.setup( self.pins[2], GPIO.OUT )
        GPIO.setup( self.pins[3], GPIO.OUT )

        # initializing
        GPIO.output( self.pins[0], GPIO.LOW )
        GPIO.output( self.pins[1], GPIO.LOW )
        GPIO.output( self.pins[2], GPIO.LOW )
        GPIO.output( self.pins[3], GPIO.LOW )

    def runMotor(self, degrees, direction):
        gpio_lock = threading.Lock()

        # careful lowering this, at some point you run into the mechanical limitation of how quick your motor can move
        step_sleep = 0.002

        step_count = 4096 / 360 * degrees # 5.625*(1/64) per step, 4096 steps is 360°
        int_step_count = int(step_count)

        # defining stepper motor sequence (found in documentation http://www.4tronix.co.uk/arduino/Stepper-Motors.php)
        step_sequence = [[1,0,0,1],
                        [1,0,0,0],
                        [1,1,0,0],
                        [0,1,0,0],
                        [0,1,1,0],
                        [0,0,1,0],
                        [0,0,1,1],
                        [0,0,0,1]]


        motor_step_counter = 0

        # the meat
        try:

            for i in range(int_step_count):
                for pin in range(0, len(self.pins)):
                    with gpio_lock:
                        GPIO.output(self.pins[pin], step_sequence[motor_step_counter][pin] )
                motor_step_counter = (motor_step_counter - 1) % 8 if direction == True else (motor_step_counter + 1) % 8
                time.sleep(step_sleep)

        except KeyboardInterrupt:
            exit(1)

    def cleanup(self):
        GPIO.output( self.pins[0], GPIO.LOW )
        GPIO.output( self.pins[1], GPIO.LOW )
        GPIO.output( self.pins[2], GPIO.LOW )
        GPIO.output( self.pins[3], GPIO.LOW )
        GPIO.cleanup()
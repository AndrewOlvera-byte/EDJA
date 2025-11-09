from MotorManager import MotorManager
import threading
import RPi.GPIO as GPIO
class App:
    def __init__(self):
        self.gpio_pins_up_down = [17, 18, 27, 22]
        self.up_down_motor = MotorManager(self.gpio_pins_up_down)

        self.gpio_pins_right_left = [23,24,25,5]
        self.right_left_motor = MotorManager(self.gpio_pins_right_left)

    def run(self):
        thread1 = threading.Thread(target=self.up_down_motor.runMotor, args=(360.0, False))
        thread2 = threading.Thread(target=self.right_left_motor.runMotor, args=(360.0, False))

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        self.up_down_motor.cleanup()
        self.right_left_motor.cleanup()

        GPIO.cleanup()
        exit(0)


    
if __name__ == "__main__":
    app = App()
    app.run()
    
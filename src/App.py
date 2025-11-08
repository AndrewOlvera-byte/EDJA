from MotorManager import MotorManager
class App:
    def __init__(self):
        self.gpio_pins_up_down = [17, 18, 27, 22]
        self.up_down_motor = MotorManager(self.gpio_pins_up_down)

        #self.gpio_pins_right_left = [23,24,25,5]
        #self.right_left_motor = MotorManager(self.gpio_pins_right_left)

    def run(self):
        self.up_down_motor.runMotor(360.0, True)
        #self.right_left_motor.runMotor(360.0, True)

        #TODO: Implement the main loop of the application

    
if __name__ == "__main__":
    app = App()
    app.run()
    
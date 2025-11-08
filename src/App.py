from MotorManager import MotorManager
class App:
    def __init__(self):
        self.motor = MotorManager(360.0)

    def run(self):
        self.motor.runMotor()
        #TODO: Implement the main loop of the application

    
if __name__ == "__main__":
    app = App()
    app.run()
    
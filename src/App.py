class App:
    def __init__(self):
        self.vis = VisionManager()
        self.track = TrackAlgo()
        self.motor = MotorManager()

    def run(self):
        motor.runMotor(360)
        #TODO: Implement the main loop of the application
    
    
if __name__ == "__main__":
    motor = App()
    motor.runMotor(360)
    
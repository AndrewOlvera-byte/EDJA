class App:
    def __init__(self):
        self.vis = VisionManager()
        self.track = TrackAlgo()
        self.motor = MotorManager()

    def run(self):
        #TODO: Implement the main loop of the application
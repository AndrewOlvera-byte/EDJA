import math
import numpy as np
class TrackAlgo:
    def __init__(self,hypotenuse, angle):
        self.hypotenuse = hypotenuse
        self.angleRad = math.radians(angle)

    def runAlgo(self):
        change_in_x = self.hypotenuse * np.cos(self.angleRad)
        change_in_y = self.hypotenuse * np.sin(self.angleRad)
        results = [change_in_x, change_in_y]
        return results


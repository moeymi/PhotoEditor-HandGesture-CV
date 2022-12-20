import cv2 as cv

class gesture_estimator:


    def __init__(self):
        self.threshold = 0.05
        self.gestures = {
            'zero':     0.83,
            'one':      0.75,
            'two':      0.60,
            'three':    0.6,
            'four':     0.55,
            'five':     0.45
        }
    
    def estimate_from_tips(self, value):
        if abs(value - self.gestures['zero']):
            return 'zero'
        return 'unknown'
    
    def estimate_from_area(self,area):
        for key, value in self.gestures.items():
            if abs(area - value) < self.threshold:
                return key
        return 'unknown'
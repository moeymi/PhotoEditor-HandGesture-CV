import cv2 as cv

class gesture_estimator:
    def __init__(self):
        self.threshold = 0.05
        self.gestures = {
            'zero':     0.85,
            'one':      0.75,
            'two':      0.60,
            'three':    0.6,
            'four':     0.55,
            'five':     0.45
        }
    
    def estimate_from_tips(self, tips):
        if abs(tips - self.gestures['zero']):
            return 'zero'
        return 'unknown'
    
    def estimate_from_area(self, area):
        for key, value in self.gestures.items():
            if abs(area - value) < self.threshold:
                return key
        return 'unknown'
    
    def estimate_from_both(self, area, tips_cnt):
        if area >= self.gestures['zero']:
            return 'zero'
        
        elif tips_cnt == 0:
            return 'one'
        
        elif tips_cnt == 1:
            return 'two'
        
        elif tips_cnt == 2:
            return 'three'
        
        elif tips_cnt == 3:
            return 'four'
        
        elif tips_cnt == 4:
            return 'five'
            
import cv2 as cv
import numpy as np
import timeit



#Your statements here

stop = timeit.default_timer()

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
        
        self.__previous_gestures = []
        
        self.frame_counter = 0
        
        self.predicted_gesture = 'zero'
        
        self.timer  = timeit.default_timer()
    
    def estimate_from_tips(self, tips):
        if abs(tips - self.gestures['zero']):
            return 'zero'
        return 'unknown'
    
    def estimate_from_area(self, area):
        for key, value in self.gestures.items():
            if abs(area - value) < self.threshold:
                return key
        return 'unknown'
    
    def estimate(self, area, defect_count):
        self.frame_counter += 1
        
        gesture = 0
        
        if area >= self.gestures['zero']:
            gesture = 0
        
        elif defect_count == 0:
            gesture = 1
        
        elif defect_count == 1:
            gesture = 2
        
        elif defect_count == 2:
            gesture =  3
        
        elif defect_count == 3:
            gesture =  4
        
        elif defect_count == 4:
            gesture =  5
            
        self.__previous_gestures.append(gesture)
        
        if timeit.default_timer() - self.timer >= 2.5:
            self.timer = timeit.default_timer() 
            self.predicted_gesture = np.bincount(np.array(self.__previous_gestures)).argmax()
            self.__previous_gestures = [gesture]
            
        return gesture
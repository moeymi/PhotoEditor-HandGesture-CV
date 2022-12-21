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
            'two':      0.68,
            'three':    0.6,
            'four':     0.55,
            'five':     0.45
        }
        
        self.__previous_gestures = []
        
        self.frame_counter = 0
        
        self.predicted_gesture = 0
        
        self.timer  = timeit.default_timer()
    
    def estimate(self, area, defect_count, farthest_point, center_point):
        self.frame_counter += 1
        
        gesture = 'zero'
        
        if area >= self.gestures['zero']:
            gesture = 'zero'
        
        elif defect_count == 0:
            if farthest_point[1] >= center_point[1]:
                gesture = 'save'
            else:
                gesture = 'translate'
        
        elif defect_count == 1:
            if area >= self.gestures['two']:
                gesture = 'rotate'
            else:
                gesture = 'six'
        
        elif defect_count == 2:
            gesture =  'scale'
        
        elif defect_count == 3:
            gesture =  'draw'
        
        elif defect_count == 4:
            gesture =  'erase'
            
        self.__previous_gestures.append(gesture)
        
        if timeit.default_timer() - self.timer >= 2.5:
            self.timer = timeit.default_timer() 
            self.predicted_gesture = max(set(self.__previous_gestures), key=self.__previous_gestures.count)
            self.__previous_gestures = [gesture]
            
        return gesture
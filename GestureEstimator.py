import cv2 as cv
import numpy as np
import timeit



#Your statements here

stop = timeit.default_timer()

class gesture_estimator:
    def __init__(self):
        self.threshold = 0.05
        self.gestures = {
            'zero':     0.89,
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
            if center_point[1] - farthest_point[1] >= 40:
                gesture = 'translate'
            elif farthest_point[0] - center_point[0] >= 40:
                gesture = 'brush_size'
            elif center_point[0] - farthest_point[0] >= 40:
                gesture = 'color_pick'
            else:
                gesture = 'save'
        
        elif defect_count == 1:
            if area >= self.gestures['two']:
                gesture = 'rotate'
            else:
                gesture = 'skew'
        
        elif defect_count == 2:
            gesture =  'scale'
        
        elif defect_count == 3:
            gesture =  'draw'
        
        elif defect_count == 4:
            gesture =  'erase'
            
        self.__previous_gestures.append(gesture)
        
        if timeit.default_timer() - self.timer >= 1.75:
            self.timer = timeit.default_timer() 
            self.predicted_gesture = max(set(self.__previous_gestures), key=self.__previous_gestures.count)
            self.__previous_gestures = [gesture]
            
        return gesture
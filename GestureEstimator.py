import cv2 as cv

class gesture_estimator:
    
    gestures = {
        'fist': 0,
        'fingers_spread': 1,
        'thumb_up': 2,
        'thumb_down': 3
    }

    
    def __init__(self):
        None
    
    def estimate_from_tips(self, tips):
        gesture = 'unknown'
        if tips is not None:
            if tips.shape[0] == 0:
                gesture = 'fist'
            elif tips.shape[0] == 1:
                gesture = 'fingers_spread'
            elif tips.shape[0] > 2:
                gesture = 'thumb_up'
            else:
                gesture = 'thumb_down'
        return gesture
    
    def estimate_from_area(area):
        None
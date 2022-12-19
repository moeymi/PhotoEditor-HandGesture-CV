import cv2 as cv
import numpy as np

class photo_editor:
    
    def __init__(self):
        self.img = None
        
    def translate(self, img, vec):
        rows,cols,_ = img.shape
        M = np.float32([[1,0,vec[0]],[0,1,vec[1]]])
        dst = cv.warpAffine(img,M,(cols,rows))
        return dst
        
    
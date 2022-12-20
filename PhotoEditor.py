import cv2 as cv
import numpy as np
from skimage import io
from skimage import transform as tf #to import this lib , pip install scikit-image

class photo_editor:
    
    def __init__(self):
        self.img = None
        
    def translate(self, img, vec):
        rows,cols,_ = img.shape
        M = np.float32([[1,0,vec[0]],[0,1,vec[1]]])
        dst = cv.warpAffine(img,M,(cols,rows))
        return dst

    def rotate(self,img,angle):
        (h,w) = img.shape[:2]
        center = (w/2,h/2)
        M = cv.getRotationMatrix2D(center=center,angle=angle,scale=1.0)
        rotated = cv.warpAffine(img, M, (w, h))
        return rotated
    
    def scale(self,img,scale_percentage):
        width = int(img.shape[1] * scale_percentage / 100)
        height = int(img.shape[0] * scale_percentage / 100)
        dim = (width, height)
        resized = cv.resize(img,dim)
        return resized
    
    def skew(self,img,shear):
        afine_tf = tf.AffineTransform(shear=shear)
        modified = tf.warp(image=img,inverse_map=afine_tf)
        return modified

    


        
    
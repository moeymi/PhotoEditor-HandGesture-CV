import cv2 as cv
import numpy as np
from skimage import io
from skimage import transform as tf #to import this lib , pip install scikit-image

class photo_editor:
    
    def __init__(self):
        self.no_drawing_image = None
        self.current_image = None
        self.cur_color = [255, 255, 255]
        self.brush_size = 10
        self.brush_color = (255, 0, 255)
        
    def translate(self, img, normalized_vec):
        rows,cols,_ = img.shape
        
        self.current_image = img
        
        trans_vec = (self.current_image.shape[1] * normalized_vec[0], normalized_vec[1] * self.current_image.shape[0])
        
        M = np.float32([[1,0,trans_vec[0]],[0,1,trans_vec[1]]])
        dst = cv.warpAffine(img,M,(cols,rows))
        
        self.current_image = dst.copy()
        
        return self.current_image

    def rotate(self, img, angle):
        (h,w) = img.shape[:2]
        center = (w/2,h/2)
        M = cv.getRotationMatrix2D(center=center,angle=angle,scale=1.0)
        rotated = cv.warpAffine(img, M, (w, h))
        
        self.current_image = rotated
        
        return self.current_image
    
    def scale(self,img,scale_percentage_x ,scale_percentage_y ):
        width = int(img.shape[1] * scale_percentage_x)
        height = int(img.shape[0] * scale_percentage_y)
        dim = (width, height)
        resized = cv.resize(img,dim,interpolation = cv.INTER_AREA)
        cv.imshow("resize" ,  resized)
        self.current_image = resized
        
        return self.current_image
    
    def skew(self,img,shear):
        afine_tf = tf.AffineTransform(shear=shear)
        modified = tf.warp(image=img,inverse_map=afine_tf)
        return modified
    
    def draw(self, img, point):
        self.current_image = img
            
        cv.circle(self.current_image, point, self.brush_size, self.brush_color, -1)
        return self.current_image
    
    def erase(self, img, no_draw_img, point):
        self.current_image = img
        
        point = tuple(x - self.brush_size for x in point)
        
        crop_original = no_draw_img[point[1]:(point[1] + 2 * self.brush_size), point[0]:(point[0]+ 2 * self.brush_size)]
        
        self.current_image[point[1]:(point[1] + 2 * self.brush_size), point[0]:(point[0]+ 2 * self.brush_size)] = crop_original
        
        return self.current_image
    
    def scale_brush (self , scale) :
        self.brush_size *= scale
    
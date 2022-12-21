import cv2
import numpy as np
import keyboard
import regex as re

from HandTracking import hand_tracker
from GestureEstimator import gesture_estimator
from PhotoEditor import photo_editor
from PhotoEditorGUI import App_GUI
from Utils import hand_helper

class Runner:
    def __init__(self):
        self.options = {
            'idle': 0,
            'translating': 1,
            'rotating': 2,
            'scale': 3
        }
        
        self.kernel_size = 8
        self.kernel = np.ones((self.kernel_size,self.kernel_size),np.uint8)
        self.iters = 2
        
        self.threshold = 150

        self.window_name = 'Photo Editor'

        self.camera_frame_width = 600 # Height is auto calculated based on aspect ratio
        self.camera_frame_x_offset = self.camera_frame_y_offset = 20

        self.ge = self.pe = self.ht = self.hh = None

        self.start_center = 0
        self.cursor_pos = (0,0)
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        self.gui = App_GUI()
        self.ge = gesture_estimator()
        self.pe = photo_editor()
        self.hh = hand_helper()

    def render_window(self, camera_frame):
            if keyboard.is_pressed("v"):
                camera_frame = self.ht.substraction_mask
                
            asp_ratio = camera_frame.shape[0] / camera_frame.shape[1]
            camera_frame_height = int(self.camera_frame_width * asp_ratio)
            camera_frame = cv2.resize(camera_frame, (self.camera_frame_width, camera_frame_height))
            
            show_img = self.editted_image.copy()
            
                
            show_img[self.camera_frame_y_offset:self.camera_frame_y_offset+camera_frame.shape[0], self.camera_frame_x_offset:self.camera_frame_x_offset+camera_frame.shape[1]] = camera_frame

            self.cursor_pos = (int((self.ht.hand_center[0] / camera_frame.shape[1]) * self.editted_image.shape[1]), 
                int((self.ht.hand_center[1]  / camera_frame.shape[0])* self.editted_image.shape[0]))  
            cv2.circle(show_img, self.cursor_pos, 10, self.pe.brush_color, thickness=2, lineType=8, shift=0)

            cv2.imshow(self.window_name, show_img)
            cv2.waitKey(1)

    def get_cameras_indices(self):
        index = 0
        arr = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                arr.append(index)
            cap.release()
            index += 1
        return arr

    def process_editting_input(self, camera_frame):   
        if not self.ge.is_clicked:
            return     
        if self.ge.predicted_gesture == 1:
            
            if self.start_center == 0:
                self.start_center = self.ht.hand_center
            
            trans_vec = self.hh.calculate_translation_normalized(self.start_center, self.ht.hand_center, camera_frame.shape)
            self.editted_image = self.org_img
            self.editted_image = self.pe.translate(self.editted_image, trans_vec)
            
        elif self.ge.predicted_gesture == 2:
            self.editted_image = self.org_img
            self.editted_image = self.pe.draw(self.editted_image, self.cursor_pos)
            
        elif self.ge.predicted_gesture == 3:
            self.editted_image = self.org_img
            self.editted_image = self.pe.erase(self.editted_image, self.cursor_pos)
        else:
            self.org_img = self.editted_image
            self.start_center = 0
            
    def load_config(self):
        image_dir, camera_ind = self.gui.load(self.get_cameras_indices())
        camera_ind = int(re.sub('\D', '', camera_ind))
        
        self.capture = cv2.VideoCapture(camera_ind)
        
        if not self.capture.isOpened() or image_dir == "":
            return False
        
        _, frame = self.capture.read()
        
        self.ht = hand_tracker(frame)
        self.org_img = cv2.imread(image_dir)
        
        return True
            
    def draw_gizmos(self, camera_frame):
        #ht.draw_farthestpoint(frame, [0, 255, 255])
        
        self.ht.draw_tips(camera_frame, [255, 0, 255])
        self.ht.draw_convex_hull(camera_frame)
        self.ht.draw_contours(camera_frame)
        
        cv2.putText(camera_frame, str(self.ge.predicted_gesture) ,(30, 100), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255) , 2, cv2.LINE_AA)
        cv2.putText(camera_frame, "IS CLICKED = " + str(self.ge.is_clicked) ,(10, 130), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)
        #self.ht.draw_farthestpoint(camera_frame, [255, 0, 255])
        
    def process_tuning_input(self):
        if keyboard.is_pressed("1"):
            
            self.kernel_size += 1
            self.kernel = np.ones((self.kernel_size,self.kernel_size),np.uint8)
        elif keyboard.is_pressed("2"):
            self.kernel_size -= 1
            if self.kernel_size <= 0:
                self.kernel_size = 0
            self.kernel = np.ones((self.kernel_size,self.kernel_size),np.uint8)
            
        if keyboard.is_pressed("3"):
            self.threshold += 1
        elif keyboard.is_pressed("4"):
            self.threshold -= 1
            
        if keyboard.is_pressed("5"):
            self.iters += 1
        elif keyboard.is_pressed("6"):
            self.iters -= 1
            if self.iters <= 0:
                self.iters = 0

    def main(self):
        
        self.camera_frame = self.capture.read()
        self.editted_image = self.org_img
        
        _, camera_frame = self.capture.read()
        camera_frame = cv2.flip(camera_frame, 1)
        
        # Reset
        if keyboard.is_pressed("r"):
            self.ht = hand_tracker(camera_frame)
            
        # Process frame
        elif self.ht.is_hand_hist_created and self.ht.bgFrame is not None:
            self.process_tuning_input()
            
            if self.ht.process(camera_frame, self.kernel, self.threshold, self.iters):
                self.draw_gizmos(camera_frame)
                self.process_editting_input(camera_frame)
                self.ge.estimate(self.ht.area_average_percentage, len(self.ht.tips))
            
        # Prepare calibration
        else:
            if self.ht.bgFrame is not None:
                self.ht.draw_rect(camera_frame)
                if keyboard.is_pressed("z"):
                    self.ht.calculate_hand_histogram(camera_frame)
            else:
                self.ht.show_save_bg(camera_frame)
                if keyboard.is_pressed("s"):
                    self.ht.saveBg_frame(camera_frame)
                    
        self.render_window(camera_frame)

    def destroy(self):
        cv2.destroyAllWindows()
        self.capture.release()

def main():
    runner = Runner()
    if not runner.load_config():
        return
    while runner.capture.isOpened():
        cv2.waitKey(1)
        runner.main()
        
        if keyboard.is_pressed("ESC"):
            break
        
    runner.destroy()
    
if __name__ == '__main__':
    main()

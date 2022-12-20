import cv2
import numpy as np
import keyboard

from HandTracking import hand_tracker
from GestureEstimator import gesture_estimator
from PhotoEditor import photo_editor
from PhotoEditorGUI import App_GUI
from Utils import hand_helper

options = {
    'idle': 0,
    'translating': 1,
    'rotating': 2,
    'scale': 3
}

window_name = 'Photo Editor'

camera_frame_width = 600 # Height is auto calculated based on aspect ratio
camera_frame_x_offset=camera_frame_y_offset=20

def main():
    gui = App_GUI()
    
    dr = gui.load_file()
    
    org_img = cv2.imread(dr)
    
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        return
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    _, camera_frame = capture.read()
    
    asp_ratio = camera_frame.shape[0] / camera_frame.shape[1]
    
    start_center = 0
    
    ge = gesture_estimator()
    pe = photo_editor()
    ht = hand_tracker(camera_frame)
    hh = hand_helper()
    
    editted_image = org_img
    while capture.isOpened():
        
        pressed_key = cv2.waitKey(1)
        _, camera_frame = capture.read()
        
        camera_frame = cv2.flip(camera_frame, 1)

        # Reset
        if keyboard.is_pressed("r"):
            ht = hand_tracker(camera_frame)
            
        # Process frame
        elif ht.is_hand_hist_created:
            ht.process(camera_frame, interpolate = False)
            
            #ht.draw_farthestpoint(frame, [0, 255, 255])
            ht.draw_tips(camera_frame, [255, 0, 255])
            ht.draw_convex_hull(camera_frame)
            ht.draw_contours(camera_frame)
            ht.draw_farthestpoint(camera_frame, [255, 0, 255])
            #print(ge.estimate_from_tips(ht.tips))
            
        # Calibrate skin histogram
        elif keyboard.is_pressed("z"):
            ht.calculate_hand_histogram(camera_frame)
            
        # Prepare calibration
        else:
            ht.draw_rect(camera_frame)

        if keyboard.is_pressed("s"):
            ht.saveBg_frame(camera_frame)

        if keyboard.is_pressed("v"):
            cv2.imshow("subtractor", cv2.subtract(camera_frame , ht.bgFrame ))
            
        if keyboard.is_pressed("f"):
            if start_center == 0:
                start_center = ht.hand_center
            
            trans_vec = hh.calculate_translation_normalized(start_center, ht.hand_center, camera_frame.shape)
            editted_image = org_img
            editted_image = pe.translate(editted_image, np.multiply(trans_vec,org_img.shape[0]).astype(int))
        else:
            org_img = editted_image
            start_center = 0
        
        camera_frame_height = int(camera_frame_width * asp_ratio)
        camera_frame = cv2.resize(camera_frame, (camera_frame_width, camera_frame_height))
        
        show_img = editted_image.copy()
        
        show_img[camera_frame_y_offset:camera_frame_y_offset+camera_frame.shape[0], camera_frame_x_offset:camera_frame_x_offset+camera_frame.shape[1]] = camera_frame

        if keyboard.is_pressed("ESC"):
            break
        
            
        cv2.imshow(window_name, show_img)
        pressed_key = cv2.waitKey(1) # force refreshing the window IT HAS TO BE THE LAST THING IN THE LOOP

    cv2.destroyAllWindows()
    capture.release()


if __name__ == '__main__':
    main()
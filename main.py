import cv2
import numpy as np
import keyboard
import regex as re

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

def init_window():
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
def render_window(camera_frame, img):
    
        asp_ratio = camera_frame.shape[0] / camera_frame.shape[1]
        camera_frame_height = int(camera_frame_width * asp_ratio)
        camera_frame = cv2.resize(camera_frame, (camera_frame_width, camera_frame_height))
        
        show_img = img.copy()
        show_img[camera_frame_y_offset:camera_frame_y_offset+camera_frame.shape[0], camera_frame_x_offset:camera_frame_x_offset+camera_frame.shape[1]] = camera_frame

        cv2.imshow(window_name, show_img)
        cv2.waitKey(1)

def get_cameras_indices():
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

def main():
    
    gui = App_GUI()
    
    dr, ind = gui.load(get_cameras_indices())
    
    ind = int(re.sub('\D', '', ind))
    
    capture = cv2.VideoCapture(ind)
    
    if not capture.isOpened() or dr == "":
        return
    
    init_window()
    org_img = cv2.imread(dr)
    _, camera_frame = capture.read()
    
    ge = gesture_estimator()
    pe = photo_editor()
    ht = hand_tracker(camera_frame)
    hh = hand_helper()
    
    start_center = 0
    
    editted_image = org_img
    while capture.isOpened():
        cv2.waitKey(1)
        _, camera_frame = capture.read()
        
        camera_frame = cv2.flip(camera_frame, 1)

        # Reset
        if keyboard.is_pressed("r"):
            ht = hand_tracker(camera_frame)
            
        # Process frame
        elif ht.is_hand_hist_created:
            ht.process(camera_frame, interpolate = False)
            cv2.putText(camera_frame, "average : " + ge.estimate_from_area(ht.calculateAverageValue()), (0, 250), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255) , 2, cv2.LINE_AA)
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
            cv2.imshow("subtractor", ht.subtractBackgroundFromFrame(camera_frame,150))
            
        if keyboard.is_pressed("f"):
            if start_center == 0:
                start_center = ht.hand_center
            
            trans_vec = hh.calculate_translation_normalized(start_center, ht.hand_center, camera_frame.shape)
            trans_vec = np.multiply(trans_vec,org_img.shape[0]).astype(int)
            editted_image = org_img
            editted_image = pe.translate(editted_image, trans_vec)
        else:
            org_img = editted_image
            start_center = 0

        if keyboard.is_pressed("ESC"):
            break
        
        render_window(camera_frame, editted_image)

    cv2.destroyAllWindows()
    capture.release()

if __name__ == '__main__':
    main()
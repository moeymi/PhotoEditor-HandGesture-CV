import cv2
import numpy as np

import HandTracking

def main():
    
    capture = cv2.VideoCapture(0)
    
    
    if not capture.isOpened():
        return
    
    _, frame = capture.read()
    frame = cv2.flip(frame, 1)
    
    hand_tracker = HandTracking.hand_tracker(frame)
    
    while capture.isOpened():
        
        pressed_key = cv2.waitKey(1)
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)

        if hand_tracker.is_hand_hist_created:
            hand_tracker.process(frame)
            
        elif pressed_key & 0xFF == ord('z'):
            hand_tracker.calculate_hand_histogram(frame)
        else:
            hand_tracker.draw_rect(frame)

        cv2.imshow("Live Feed", hand_tracker.rescale_frame(frame))

        if pressed_key == 27:
            break

    cv2.destroyAllWindows()
    capture.release()


if __name__ == '__main__':
    main()
import cv2 as cv
import numpy as np
class hand_tracker:
    
    def __init__(self, frame):
        self.total_rectangle = 9
        self.is_hand_hist_created = False
        self.is_bg_frame_taken = False  
        
        self.far_points_history = []
        
        self.hand_center = 0
        
        self.__rows, self.__cols, _ = frame.shape

        self.area_percentages = []
        self.averaging_fps_threshold = 5
        self.area_average_percentage = 0
        
        self._last_frame_counter = 0
        
        self.bgFrame = None
        
        self.hand_rect_one_x = np.array(
            [6 * self.__rows / 20, 6 * self.__rows / 20, 6 * self.__rows / 20, 9 * self.__rows / 20, 9 * self.__rows / 20, 9 * self.__rows / 20, 12 * self.__rows / 20,
            12 * self.__rows / 20, 12 * self.__rows / 20], dtype=np.uint32)

        self.hand_rect_one_y = np.array(
            [9 * self.__cols / 20, 10 * self.__cols / 20, 11 * self.__cols / 20, 9 * self.__cols / 20, 10 * self.__cols / 20, 11 * self.__cols / 20, 9 * self.__cols / 20,
            10 * self.__cols / 20, 11 * self.__cols / 20], dtype=np.uint32)
        
    
    # Drawing
    def draw_farthestpoint(self, frame, color):
        cv.circle(frame, self.hand_center, 10, color, 2)
        
        if self.far_points_history is not None:
            for i in range(len(self.far_points_history)):
                cv.circle(frame, self.far_points_history[i], int(5 - (5 * i * 3) / 100), color, -1)

    def draw_convex_hull(self,frame):
        hull = [cv.convexHull(self.hand_contour)]
        cv.drawContours(frame,hull,-1,(255,255,255))
        
    def draw_contours(self,frame):
        cv.drawContours(frame,[self.hand_contour],-1,(255,255,0), 3)

    def draw_rect(self, frame):
        self.hand_rect_two_x = self.hand_rect_one_x + 10
        self.hand_rect_two_y = self.hand_rect_one_y + 10

        for i in range(self.total_rectangle):
            cv.rectangle(frame, (self.hand_rect_one_y[i], self.hand_rect_one_x[i]),
                        (self.hand_rect_two_y[i], self.hand_rect_two_x[i]),
                        (0, 255, 0), 1)
            
    def show_save_bg(self, frame):
        cv.putText(frame, "Press S to save background", (100, 50), cv.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv.LINE_AA)
            
    def rescale_frame(self, frame, wpercent=130, hpercent=130):
        width = int(frame.shape[1] * wpercent / 200)
        height = int(frame.shape[0] * hpercent / 200)
        return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)

    def saveBg_frame(self , frame):
        self.bgFrame = frame


    def __get_contours(self, hist_mask_image):
        gray_hist_mask_image = cv.cvtColor(hist_mask_image, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray_hist_mask_image, 0, 255, 0)
        cont, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        return cont
            
    def __calculateAverageValue(self):
        if len(self.area_percentages) > 0:
            self.area_average_percentage = np.sum(self.area_percentages) / len(self.area_percentages)
        
    def __clear_area_percentages(self):
        if len(self.area_percentages) > 0:
            self.area_percentages = [self.area_percentages[len(self.area_percentages) - 1]]

    def calculate_hand_histogram(self, frame):
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

        for i in range(self.total_rectangle):
            roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[self.hand_rect_one_x[i]:self.hand_rect_one_x[i] + 10,
                                            self.hand_rect_one_y[i]:self.hand_rect_one_y[i] + 10]

        self.hand_hist = cv.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        self.hand_hist = cv.normalize(self.hand_hist, self.hand_hist, 0, 255, cv.NORM_MINMAX)
        
        self.is_hand_hist_created = True

    def subtractBackgroundFromFrame(self , fg_frame, kernel = (7,7), threshold = 150, iters = 1):
        
        fg_frame_gray = cv.cvtColor(fg_frame, cv.COLOR_BGR2GRAY)
        bg_frame_gray = cv.cvtColor(self.bgFrame, cv.COLOR_BGR2GRAY)

        newFrame = np.square(fg_frame_gray - bg_frame_gray)
        
        _,thresh = cv.threshold(newFrame,threshold,255,cv.THRESH_BINARY)
        
        thresh=cv.morphologyEx(thresh,cv.MORPH_CLOSE, kernel, iterations=iters)
        thresh=cv.morphologyEx(thresh,cv.MORPH_OPEN, kernel)
        
        thresh = cv.merge((thresh, thresh, thresh))
        
        return cv.bitwise_and(fg_frame, thresh)
        


    def __get_hist_mask(self, frame, hist):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        dst = cv.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

        disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (31, 31))
        cv.filter2D(dst, -1, disc, dst)

        _, thresh = cv.threshold(dst, 150, 255, cv.THRESH_BINARY)
        
        thresh = cv.erode(thresh, (3,3), iterations=4)
        
        # thresh = cv.dilate(thresh, None, iterations=5)

        thresh = cv.merge((thresh, thresh, thresh))

        return cv.bitwise_and(frame, thresh)


    def __get_centroid(self, max_contour):
        moment = cv.moments(max_contour)
        if moment['m00'] != 0:
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])
            return cx, cy
        else:
            return None


    def __get_farthest_point(self, defects, contour, centroid):
        if defects is not None and centroid is not None:
            s = defects[:, 0][:, 0]
            cx, cy = centroid
            
            points = np.array([contour[point] for point in s if contour[point][0][1] <= cy])
            points = points[:, 0, :]
            
            x = np.array(points[:, 0], dtype=np.float)
            y = np.array(points[:, 1], dtype=np.float)
    
            xp = cv.pow(cv.subtract(x, cx), 2)
            yp = cv.pow(cv.subtract(y, cy), 2)
            
            kk = cv.add(xp, yp)
            
            dist = cv.sqrt(kk)

            dist_max_i = np.argmax(dist)
            
            if dist_max_i < len(s):
                farthest_defect = s[dist_max_i]
                farthest_point = tuple(contour[farthest_defect][0])
                return farthest_point
            else:
                return None
            
    def __get_tips(self, defects, contours):
        if defects is not None:
            cnt = 0
            tips = []
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(contours[s][0])
                end = tuple(contours[e][0])
                far = tuple(contours[f][0])
                a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  #      cosine theorem
                if angle <= np.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    tips.append(start)
            tips = np.array(tips)
            if cnt > 0:
                cnt = cnt+1
            return tips, cnt
        
        return None, None
        
    def draw_tips(self, frame, color):
        if self.tips is not None:
            for tip in self.tips:
                cv.circle(frame, tip, 4, color, -1)
        
            #cv.putText(frame, "Tip count : " + str(self.tip_cnt), (0, 50), cv.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv.LINE_AA)

    def process(self, frame, kernel, threshold, iters, interpolate=True):
        
        self._last_frame_counter +=1
        
        self.substraction_mask = self.subtractBackgroundFromFrame(frame, kernel=kernel, threshold=threshold, iters = iters)
        self.hist_mask_image = self.__get_hist_mask(self.substraction_mask, self.hand_hist)

        self.hist_mask_image = cv.erode(self.hist_mask_image, None, iterations=2)
        self.hist_mask_image = cv.dilate(self.hist_mask_image, None, iterations=2)

        self.contour_list = self.__get_contours(self.hist_mask_image)
        
        if len(self.contour_list) <= 0:
            return False
        
        self.hand_contour = max(self.contour_list, key=cv.contourArea)
        
        self.hand_center = self.__get_centroid(self.hand_contour)
            
        if self.hand_contour is not None:
            hull = cv.convexHull(self.hand_contour, returnPoints=False)
            defects = cv.convexityDefects(self.hand_contour, hull)
            
            contour_area = cv.contourArea(self.hand_contour)
            hull_area = cv.contourArea(cv.convexHull(self.hand_contour))

            self.area_percentages.append(contour_area/hull_area)
            
            if self._last_frame_counter >= self.averaging_fps_threshold:
                self.__calculateAverageValue()
                self.__clear_area_percentages()
            
            cv.putText(frame, "current : " + str(contour_area/hull_area), (0, 50), cv.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255) , 2, cv.LINE_AA)
            cv.putText(frame, "average : " + str(self.area_average_percentage), (0, 200), cv.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255) , 2, cv.LINE_AA)
           

            #self.far_point = self.__get_farthest_point(defects, self.hand_contour, self.hand_center)
            self.tips, self.tip_cnt = self.__get_tips(defects, self.hand_contour)
            
            """
            if len(self.far_points_history) < 10:
                self.far_points_history.append(self.far_point)
            else:
                self.far_points_history.pop(0)
                self.far_points_history.append(self.far_point)
                """
            
            return True
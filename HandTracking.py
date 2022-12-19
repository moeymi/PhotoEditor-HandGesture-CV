import cv2 as cv
import numpy as np
class hand_tracker:
    
    def __init__(self, frame):
        self.total_rectangle = 9
        self.is_hand_hist_created = False
        
        self.traverse_point = []
        
        self.contour_area = 0
        self.hull_area = 0
        
        self.rows, self.cols, _ = frame.shape
        
        self.hand_rect_one_x = np.array(
            [6 * self.rows / 20, 6 * self.rows / 20, 6 * self.rows / 20, 9 * self.rows / 20, 9 * self.rows / 20, 9 * self.rows / 20, 12 * self.rows / 20,
            12 * self.rows / 20, 12 * self.rows / 20], dtype=np.uint32)

        self.hand_rect_one_y = np.array(
            [9 * self.cols / 20, 10 * self.cols / 20, 11 * self.cols / 20, 9 * self.cols / 20, 10 * self.cols / 20, 11 * self.cols / 20, 9 * self.cols / 20,
            10 * self.cols / 20, 11 * self.cols / 20], dtype=np.uint32)

    def rescale_frame(self, frame, wpercent=130, hpercent=130):
        width = int(frame.shape[1] * wpercent / 200)
        height = int(frame.shape[0] * hpercent / 200)
        return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)


    def __contours(self, hist_mask_image):
        gray_hist_mask_image = cv.cvtColor(hist_mask_image, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray_hist_mask_image, 0, 255, 0)
        cont, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        return cont

    def draw_rect(self, frame):
        self.hand_rect_two_x = self.hand_rect_one_x + 10
        self.hand_rect_two_y = self.hand_rect_one_y + 10

        for i in range(self.total_rectangle):
            cv.rectangle(frame, (self.hand_rect_one_y[i], self.hand_rect_one_x[i]),
                        (self.hand_rect_two_y[i], self.hand_rect_two_x[i]),
                        (0, 255, 0), 1)


    def calculate_hand_histogram(self, frame):
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

        for i in range(self.total_rectangle):
            roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[self.hand_rect_one_x[i]:self.hand_rect_one_x[i] + 10,
                                            self.hand_rect_one_y[i]:self.hand_rect_one_y[i] + 10]

        self.hand_hist = cv.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        self.hand_hist = cv.normalize(self.hand_hist, self.hand_hist, 0, 255, cv.NORM_MINMAX)
        
        self.is_hand_hist_created = True


    def hist_masking(self, frame, hist):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        dst = cv.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

        disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (31, 31))
        cv.filter2D(dst, -1, disc, dst)

        ret, thresh = cv.threshold(dst, 150, 255, cv.THRESH_BINARY)
        
        thresh = cv.erode(thresh, (3,3), iterations=4)
        
        # thresh = cv.dilate(thresh, None, iterations=5)

        thresh = cv.merge((thresh, thresh, thresh))

        return cv.bitwise_and(frame, thresh)


    def __centroid(self, max_contour):
        moment = cv.moments(max_contour)
        if moment['m00'] != 0:
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])
            return cx, cy
        else:
            return None


    def __farthest_point(self, defects, contour, centroid):
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
            
    def __tips(self, defects, contours):
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
            if cnt > 0:
                cnt = cnt+1
            return tips, cnt
        
        return None, None
        
    def draw_tips(self, frame, color):
        for tip in self.tips:
            cv.circle(frame, tip, 4, color, -1)
        
        #cv.putText(frame, "Tip count : " + str(self.tip_cnt), (0, 50), cv.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv.LINE_AA)

    def draw_farthestpoint(self, frame, color):
        cv.circle(frame, self.cnt_centroid, 10, color, 2)
        
        if self.traverse_point is not None:
            for i in range(len(self.traverse_point)):
                cv.circle(frame, self.traverse_point[i], int(5 - (5 * i * 3) / 100), color, -1)

    def draw_convex_hull(self,frame):
        hull = [cv.convexHull(self.max_cont)]
        cv.drawContours(frame,hull,-1,(255,255,255))
        
    def draw_contours(self,frame):
        cv.drawContours(frame,[self.max_cont],-1,(255,255,0), 3)

    def process(self, frame):
        self.hist_mask_image = self.hist_masking(frame, self.hand_hist)

        self.hist_mask_image = cv.erode(self.hist_mask_image, None, iterations=2)
        self.hist_mask_image = cv.dilate(self.hist_mask_image, None, iterations=2)

        self.contour_list = self.__contours(self.hist_mask_image)
        
        if len(self.contour_list) <= 0:
            return
        
        self.max_cont = max(self.contour_list, key=cv.contourArea)
        
        self.cnt_centroid = self.__centroid(self.max_cont)

        if self.max_cont is not None:
            hull = cv.convexHull(self.max_cont, returnPoints=False)
            defects = cv.convexityDefects(self.max_cont, hull)
            
            contour_area = cv.contourArea(self.max_cont)
            hull_area = cv.contourArea(cv.convexHull(self.max_cont))
            
            if self.hull_area > 0:
                self.hull_area = (self.hull_area + hull_area) / 2
            else:
                self.hull_area = hull_area
            
            if self.contour_area > 0:
                self.contour_area = (self.contour_area + contour_area) / 2
            else:
                self.contour_area = contour_area
                
            cv.putText(frame, "current : " + str(contour_area/hull_area), (0, 50), cv.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255) , 2, cv.LINE_AA)
            cv.putText(frame, "average : " + str(self.contour_area/self.hull_area), (0, 200), cv.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255) , 2, cv.LINE_AA)
            self.far_point = self.__farthest_point(defects, self.max_cont, self.cnt_centroid)
            self.tips, self.tip_cnt = self.__tips(defects, self.max_cont)
            
            if len(self.traverse_point) < 10:
                self.traverse_point.append(self.far_point)
            else:
                self.traverse_point.pop(0)
                self.traverse_point.append(self.far_point)

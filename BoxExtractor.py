import cv2
import os
import numpy as np


class BoardExtractor:
    def __init__(self, img):
        self.image = cv2.imread(img, 0)
        self.extractedgrid = None
        self.save_img = True

    def preprocess_image(self):
        gray = self.image
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        if self.save_img:
            try:
                os.remove("StageImages/1_GaussianBlur.jpg", gray)
            except:
                pass
            cv2.imwrite("StageImages/1_GaussianBlur.jpg", gray)

        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C | cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
        if self.save_img:
            try:
                os.remove("StageImages/2_Adaptive_thresh.jpg")
            except:
                pass
            cv2.imwrite("StageImages/2_Adaptive_thresh.jpg", gray)

        gray = cv2.bitwise_not(gray)
        if self.save_img:
            try:
                os.remove("StageImages/3_bitwise.jpg")
            except:
                pass
            cv2.imwrite("StageImages/3_bitwise.jpg", gray)

        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        gray = cv2.dilate(gray, kernel)
        self.image = gray
        if self.save_img:
            try:
                os.remove("StageImages/4_dilate.jpg")
            except:
                pass
            cv2.imwrite("StageImages/4_dilate.jpg", gray)

    def detect_and_crop_grid(self):
        outerbox = self.image
        maxi = -1
        maxpt = None
        value = 10
        if self.save_img:
            try:
                os.remove("StageImages/5_outerbox.jpg")
            except:
                pass
        height, width = np.shape(outerbox)
        for y in range(height):
            row = self.image[y]
            for x in range(width):
                if row[x] >= 128:
                    area = cv2.floodFill(outerbox, None, (x, y), 64)[0]
                    if value > 0:
                        if self.save_img:
                            cv2.imwrite("StageImages/5_outerbox.jpg", outerbox)
                            value -= 1
                    if area > maxi:
                        maxpt = (x, y)
                        maxi = area

        cv2.floodFill(outerbox, None, maxpt, (255, 255, 255))

        for y in range(height):
            row = self.image[y]
            for x in range(width):
                if row[x] == 64 and x != maxpt[0] and y != maxpt[1]:
                    cv2.floodFill(outerbox, None, (x, y), 0)
        if self.save_img:
            try:
                os.remove("StageImages/6_floodFill.jpg")
            except:
                pass
            cv2.imwrite("StageImages/6_floodFill.jpg", outerbox)

        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        outerbox = cv2.erode(outerbox, kernel)
        if self.save_img:
            try:
                os.remove("StageImages/7_erode.jpg")
            except:
                pass
            cv2.imwrite("StageImages/7_erode.jpg", outerbox)

        lines = cv2.HoughLines(outerbox, 1, np.pi / 180, 200)

        def drawLine(line, img):
            height, width = np.shape(img)
            if line[0][1] != 0:
                m = -1 / np.tan(line[0][1])
                c = line[0][0] / np.sin(line[0][1])
                cv2.line(img, (0, int(c)), (width, int(m*width + c)), 255)
            else:
                cv2.line(img, (line[0][0], 0), (line[0][0], height), 255)

            return img

        tmpimg = np.copy(outerbox)
        for i in range(len(lines)):
            tmpimp = drawLine(lines[i], tmpimg)
        if self.save_img:
            try:
                os.remove("StageImages/8_drawline.jpg")
            except:
                pass
            cv2.imwrite("StageImages/8_drawline.jpg", outerbox)

        def mergeLines(lines, img):
            height, width = np.shape(img)
            for current in lines:
                if current[0][0] is None and current[0][1] is None:
                    continue
                p1 = current[0][0]
                theta1 = current[0][1]
                pt1current = [None, None]
                pt2current = [None, None]
                if (theta1 > np.pi * 45 / 180) and (theta1 < np.pi * 135 / 180):
                    pt1current[0] = 0
                    pt1current[1] = p1 / np.sin(theta1)
                    pt2current[0] = width
                    pt2current[1] = -pt2current[0] / np.tan(theta1) + p1 / np.sin(theta1)
                else:
                    pt1current[0] = p1 / np.cos(theta1)
                    pt1current[1] = 0
                    pt2current[0] = height
                    pt2current[1] = -pt2current[1] * np.tan(theta1) + p1 / np.cos(theta1)

                for pos in lines:
                    if



if __name__ == '__main__':
    path = 'test_img.jpg'
    preprocessor = BoardExtractor(path)
    preprocessor.preprocess_image()

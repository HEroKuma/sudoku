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


if __name__ == '__main__':
    path = 'test_img.jpg'
    preprocessor = BoardExtractor(path)
    preprocessor.preprocess_image()

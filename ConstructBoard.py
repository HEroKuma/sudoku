import cv2
import numpy as np
import torch
import os
import pickle
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import CNN_model

img_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

class DigitRecognizer:
    def __init__(self):
        self.writeimg = True
        self.model = CNN_model.LeNet()
        self.model.load_state_dict(torch.load('LeNet.pth'))

    def prediction(self, img):
        try:
            cleanedimg = cv2.imread(img, 0)
            cleanedimg = cv2.resize(cleanedimg, (28, 28))
        except:
            raise Exception("Img load Failed")
        self.model.eval()
        cleanedimg = img_transforms(cleanedimg).float()
        img = Variable(cleanedimg)
        img = img.unsqueeze(0)
        idx = None
        pred = self.model(img)
        _, pred = torch.max(pred.data, 1)
        print(pred.item())

    def preprocess_image(self, img):
        rows = np.shape(img)[0]
        for i in range(rows):
            cv2.floodFill(img, None, (0, i), 0)
            cv2.floodFill(img, None, (i, 0), 0)
            cv2.floodFill(img, None, (rows-i, i), 0)
            cv2.floodFill(img, None, (i, rows-i), 0)
            cv2.floodFill(img, None, (1, i), 1)
            cv2.floodFill(img, None, (i, 1), 1)
            cv2.floodFill(img, None, (rows-2, i), 1)
            cv2.floodFill(img, None, (i, rows-2), 1)

        if self.writeimg:
            try:
                os.remove("StageImages/14_floodFillCell.img")
            except:
                pass
            cv2.imwrite("StageImages/14_floodFillCell.img", img)

        rowtop = None
        rowbottom = None
        colleft = None
        colright = None
        thresholdBottom = 50
        thresholdTop = 50
        thresholdLeft = 50
        thresholdRight = 50
        center = rows // 2
        for i in range(center, rows):
            if rowbottom is None:
                temp = img[i]
                if sum(temp) < thresholdBottom or i == rows-1:
                    rowbottom = i
            if rowtop is None:
                temp = img[rows-i-1]
                if sum(temp) < thresholdTop or i == rows-1:
                    rowtop = rows-i-1
            if colright is None:
                temp = img[:, i]
                if sum(temp) < thresholdRight or i == rows-1:
                    colright = i
            if colleft is None:
                temp = img[:, rows-i-1]
                if sum(temp) < thresholdLeft or i == rows-1:
                    colleft = rows-i-1

        newimg = np.zeros(np.shape(img))
        startatX = (rows + colleft - colright) // 2
        startatY = (rows - rowbottom + rowtop) // 2
        for y in range(startatY, (rows + rowbottom - rowtop)//2):
            for x in range(startatX, (rows - colleft + colright)//2):
                newimg[y, x] = img[rowtop + y - startatY, colleft + x - startatX]

        if self.writeimg:
            try:
                os.remove("StageImages/15_bound.jpg")
            except:
                pass
            cv2.imwrite("StageImages/15_bound.jpg", newimg)
            self.writeimg = False
        return newimg


if __name__ == '__main__':
    recognizer = DigitRecognizer()
    recognizer.prediction('StageImages/cell03.jpg')

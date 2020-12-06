import cv2
import numpy as np
import torch
import os
import pickle
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import CNN_model

from BoxExtractor import *

img_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])


class DigitRecognizer:
    def __init__(self):
        self.writeimg = True
        self.model = CNN_model.MNISTResNet()
        self.model.load_state_dict(torch.load('Res.pth'))
        self.model.eval()

    def prediction(self, img):
        try:
            cleanedimg = cv2.imread(img, 0)
            cleanedimg = cv2.resize(cleanedimg, (56, 56))
        except:
            raise Exception("Img load Failed")
        cleanedimg = img_transforms(cleanedimg).float()
        img = Variable(cleanedimg)
        img = img.unsqueeze(0)
        idx = None
        pred = self.model(img)
        _, pred = torch.max(pred.data, 1)

        return pred.item()

    def preprocess_image(self, img):
        rows = np.shape(img)[0]
        for i in range(rows):
            cv2.floodFill(img, None, (0, i), 0)
            cv2.floodFill(img, None, (i, 0), 0)
            cv2.floodFill(img, None, (rows-1, i), 0)
            cv2.floodFill(img, None, (i, rows-1), 0)
            cv2.floodFill(img, None, (1, i), 1)
            cv2.floodFill(img, None, (i, 1), 1)
            cv2.floodFill(img, None, (rows-2, i), 1)
            cv2.floodFill(img, None, (i, rows-2), 1)

        if self.writeimg:
            try:
                os.remove("StageImages/14_floodFillCell.jpg")
            except:
                pass
            cv2.imwrite("StageImages/14_floodFillCell.jpg", img)

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


class ConstructGrid():
    def __init__(self, cellarray):
        self.cellarray = cellarray
        self.recognizer = DigitRecognizer()
        self.finalgrid = [[0 for _ in range(9)] for _ in range(9)]
        self.imagewritten = False

    def constructgrid(self):
        threshold = 5*255
        for i in range(9):
            for j in range(9):
                tmp = np.copy(self.cellarray[i][j])
                tmp = self.recognizer.preprocess_image(tmp)
                cv2.imwrite("BoardCells/cell"+str(i)+str(j)+".jpg", tmp)
                finsum = 0
                for k in range(28):
                    rowsum = sum(tmp[k])
                    finsum += rowsum
                if finsum < threshold:
                    self.finalgrid[i][j] = 0
                    continue
                if not self.imagewritten:
                    try:
                        os.remove("StageImages/13_cell.jpg")
                        os.remove("StageImages/14_cell_tmp.jpg")
                    except:
                        pass
                    cv2.imwrite("StageImages/13_cell.jpg", self.cellarray[i][j])
                    cv2.imwrite("StageImages/14_cell_tmp.jpg", tmp)
                pred = self.recognizer.prediction(str("BoardCells/cell"+str(i)+str(j)+".jpg"))
                print(pred)
                self.finalgrid[i][j] = int(pred)

        return self.finalgrid


if __name__ == '__main__':
    path = 'test_img.jpg'
    preprocessor = BoardExtractor(path)
    preprocessor.preprocess_image()
    preprocessor.detect_and_crop_grid()
    boardcells = preprocessor.create_image_grid()
    recognizedandconstructobj = ConstructGrid(boardcells)
    board = recognizedandconstructobj.constructgrid()
    for i in board:
        print(i)
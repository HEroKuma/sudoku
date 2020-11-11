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
        except:
            raise Exception("Img load Failed")
        self.model.eval()
        cleanedimg = img_transforms(cleanedimg).float()
        img = Variable(cleanedimg)
        img = img.unsqueeze(0)
        idx = None
        pred = self.model(img)
        _, pred = torch.max(pred.data, 1)
        print(pred)


if __name__ == '__main__':
    recognizer = DigitRecognizer()
    recognizer.prediction('StageImages/cell03.jpg')

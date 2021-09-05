import os
import sys
import cv2
import numpy as np
import torch

try:
    sys.path.append("C:/tapMobileProj/projOriginal/DAN-master/codes")  # was "..'
    from data.util import imresize
    import utils as util
except ImportError:
    pass

def checkHRidentical():
    firstImage = cv2.imread("C:/tapMobileProj/dataset/united_noVal/HR/x2/sig1.8_000008.png")
    #firstImage = cv2.imread("C:/tapMobileProj/checkHRidentical/sig1.8_000008.png")
    #firstImage = cv2.imread("C:/tapMobileProj/checkHRidentical/sig1p8_000008.png")
    secondImage = cv2.imread("C:/tapMobileProj/dataset/united_noVal/HR/x2/sig2.2_000008.png")

    diff = cv2.subtract(firstImage, secondImage)
    #print(diff[1,1,1])
    print(diff.max())
    print(np.max(diff))
    print(np.max(np.absolute(diff)))

    #firstImageTensor = util.img2tensor(firstImage)
    #secondImageTensor = util.img2tensor(secondImage)

    #if (torch.eq(firstImageTensor, secondImageTensor)):
    #    print("Well, I guess they are equal")
    #else:
    #    print("Nope, they arent the same image")
    #cv2.imwrite("C:/tapMobileProj/checkHRidentical/firstImage.png", firstImage)

    #cv2.imshow('image', firstImage)
    #cv2.waitKey(0)

if __name__ == "__main__":
    checkHRidentical()
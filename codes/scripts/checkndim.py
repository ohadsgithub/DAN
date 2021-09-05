
import numpy as np
import cv2

def checkndim():

    #data=cv2.imread("C:/tapMobileProj/checkHRidentical/sig1p8_000008.png", cv2.IMREAD_UNCHANGED)
    #data=cv2.imread("C:/tapMobileProj/checkHRidentical/sig1p8_000008.png")
    #data = cv2.imread("C:/tapMobileProj/checkHRidentical/sig1p8_000008.png")
    #print(data.ndim)

    image1 = cv2.imread("C:/tapMobileProj/testOverwrite/imageFour.png")
    # cv2.imwrite("C:/tapMobileProj/testOverwrite/imageTwo.png", image1)

    print(image1.ndim)

    #cv2.imshow('image', data)
    #cv2.waitKey(0)
    #print(data.shape)

if __name__ == "__main__":
    checkndim()
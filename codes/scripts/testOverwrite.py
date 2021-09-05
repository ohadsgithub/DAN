
import cv2


def testOverwrite():
    #image1 = cv2.imread("C:/tapMobileProj/testOverwrite/imageOne.png")
    #cv2.imwrite("C:/tapMobileProj/testOverwrite/imageTwo.png", image1)
    #image3 = cv2.imread("C:/tapMobileProj/dataset/united_noVal/source/0011.png")
    #cv2.imwrite("C:/tapMobileProj/testOverwrite/imageThree.png", image3)
    #image4 = cv2.imread("C:/tapMobileProj/testOverwrite/imageTwo.png")
    #cv2.imwrite("C:/tapMobileProj/testOverwrite/imageFour.png", image4)

    image1 = cv2.imread("C:/tapMobileProj/testOverwrite/imageFour.png")
    #cv2.imwrite("C:/tapMobileProj/testOverwrite/imageTwo.png", image1)

    print(image1.ndim)

if __name__ == "__main__":
    testOverwrite()
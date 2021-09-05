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
    #print("not imported")
    pass

def downsizeImages():
    # set parameters
    up_scale = 3
    # set data dir
    sourcedir = "C:/tapMobileProj/dataset/united_noVal/source"  # why /? was "/data/Set5/source/"
    savedir = "C:/tapMobileProj/dataset/dataset_resized"  # was "/data/Set5/"    extra / ?

    savePath = os.path.join(savedir, "x" + str(up_scale), "Source")

    if not os.path.isdir(sourcedir):
        print("Error: No source data found")
        exit(0)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    if not os.path.isdir(os.path.join(savedir, "x" + str(up_scale))):
        os.mkdir(os.path.join(savedir, "x" + str(up_scale)))

    if not os.path.isdir(savePath):
        os.mkdir(savePath)
    else:
        print("It will cover " + str(savePath))

    filepaths = sorted([f for f in os.listdir(sourcedir) if f.endswith(".png")])
    print(filepaths)
    num_files = len(filepaths)

    for i in range(num_files):
        filename = filepaths[i]
        print("No.{} -- Processing {}".format(i, filename))
        # read image
        image = cv2.imread(os.path.join(sourcedir, filename))

        width = int(np.floor(image.shape[1] / up_scale))
        height = int(np.floor(image.shape[0] / up_scale))
        # modcrop
        if len(image.shape) == 3:
            image_HR = image[0: up_scale * height, 0: up_scale * width, :]
        else:
            image_HR = image[0: up_scale * height, 0: up_scale * width]

        #img_HR = util.img2tensor(image_HR)

        image_LR = imresize(image_HR, 1 / up_scale, True)

        cv2.imwrite(os.path.join(savePath, filename), image_LR)

        #cv2.imshow('image', image_HR)
        #cv2.imshow('image', image)
        cv2.imshow('image', image_LR)
        cv2.waitKey(0)
        break



    print("Down sample Done: X" + str(up_scale))


if __name__ == "__main__":
    downsizeImages()
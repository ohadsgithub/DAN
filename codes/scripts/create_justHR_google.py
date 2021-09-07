import os
import sys
import cv2
import numpy as np
import torch

try:
    sys.path.append("/content/DAN/codes")
    from data.util import imresize
    import utils as util
except ImportError:
    pass

def create_justHR_google():
    # set parameters
    up_scale = 2
    mod_scale = 2
    # set data dir
    sourcedir = "/content/drive/MyDrive/tapmobileTestProj/trainData/data_resized/x2ds/source"
    savedir = "/content/drive/MyDrive/tapmobileTestProj/trainData/data_resized/x2HR_forLMDB" #/content/drive/MyDrive/tapmobileTestProj/trainData/data_resized/x2HR_forLMDB
    #sourcedir = "/content/drive/MyDrive/tapmobileTestProj/trainData/united_noVal/source"
    #savedir = "/content/drive/MyDrive/tapmobileTestProj/trainData/justHRforLMDB"

    saveHRpath = os.path.join(savedir, "HR", "x" + str(mod_scale))
    

    if not os.path.isdir(sourcedir):
        print("Error: No source data found")
        exit(0)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    if not os.path.isdir(os.path.join(savedir, "HR")):
        os.mkdir(os.path.join(savedir, "HR"))

    if not os.path.isdir(saveHRpath):
        os.mkdir(saveHRpath)
    else:
        print("It will cover " + str(saveHRpath))

    filepaths = sorted([f for f in os.listdir(sourcedir) if f.endswith(".png")])
    print(filepaths)
    num_files = len(filepaths)

    # kernel_map_tensor = torch.zeros((num_files, 1, 10)) # each kernel map: 1*10

    # prepare data with augementation
    
    for i in range(num_files):
        filename = filepaths[i]
        print("No.{} -- Processing {}".format(i, filename))
        # read image
        image = cv2.imread(os.path.join(sourcedir, filename))

        width = int(np.floor(image.shape[1] / mod_scale))
        height = int(np.floor(image.shape[0] / mod_scale))
        # modcrop
        if len(image.shape) == 3:
            image_HR = image[0 : mod_scale * height, 0 : mod_scale * width, :]
        else:
            image_HR = image[0 : mod_scale * height, 0 : mod_scale * width]

        cv2.imwrite(os.path.join(saveHRpath, filename), image_HR)
        

        # kernel_map_tensor[i] = ker_map
    # save dataset corresponding kernel maps
    # torch.save(kernel_map_tensor, './Set5_sig2.6_kermap.pth')
    print("HR slight change Done: X" + str(up_scale))


if __name__ == "__main__":
    create_justHR_google()

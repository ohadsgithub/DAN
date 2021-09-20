import argparse
import logging
import os
import os.path as osp
import sys
import time
from collections import OrderedDict
from glob import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr


import torch.onnx


#### options
parser = argparse.ArgumentParser()
parser.add_argument(
    "-opt",
    type=str,
    default="options/setting1/test/test_setting1_x2_small.yml",
    help="Path to options YMAL file.",
)

#parser.add_argument(
#    "-dim",
#    type=int,
#    default=1,
#    help="dim for structured prune experiment",
#)


args = parser.parse_args()
opt = option.parse(args.opt, is_train=False)

opt = option.dict_to_nonedict(opt)

#print("dim is "+str(dim))

model = create_model(opt)

#model.print_named_parameters()

dim=0 #dim=0 seems to work
model.structured_prune_experiment(dim)



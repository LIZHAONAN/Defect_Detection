# region detection by YOLO v3
# author: Zhaonan Li, zli@brandeis.edu
# created at: 4/21/2020

import os
import numpy as np
import pandas as pd
from Yolo_v3.yolo_models import *
from Yolo_v3.yolo_utils import *
import torch

import argparse

model_def = 'Yolo_v3/yolov3-custom.cfg'

parser = argparse.ArgumentParser()
parser.add_argument("--pth", '-p', type=str, required=True, help='path to weight of yolo model')
parser.add_argument("--images", type=str, required=True, help='path to image path file in .csv format')
parser.add_argument("--output", type=str, required=True, help='path to the output file')
parser.add_argument("--gpu", type=bool, default=True, help='use gpu or not')
parser.add_argument("--batch_size", '-b', type=int, default=128, help='batch size')
parser.add_argument("--conf", type=float, default=0.5, help='threshold for confidence')
parser.add_argument("--nms", type=float, default=0.5, help='threshold for non max suppression')

opt = parser.parse_args()

if opt.gpu and torch.cuda.is_available():
    print('-- cuda available, using gpu')
    device = torch.device('cuda')
    opt.gpu = True
else:
    print('-- cuda unavailable or unwanted, using cpu')
    device = torch.device('cpu')
    opt.gpu = False

print('-- loading models')
model = Darknet(model_def).to(device)
model.load_state_dict(torch.load(opt.pth, map_location=device))
print('-- finished loading')

# load img paths
df_imgs = pd.read_csv(opt.images, sep=',')

# initialize output
df_res = pd.DataFrame(columns=['path', 'class', 'x1', 'y1', 'x2', 'y2', 'confidence'])

if __name__ == '__main__':
    for paths, imgs in custom_dataloader(df_imgs, opt.batch_size):
        imgs = torch.from_numpy(imgs).unsqueeze(1).expand(-1, 3, 416, 416)
        imgs = imgs.to(device).float()
        outs = model(imgs)
        outs = non_max_suppression(outs, conf_thres=opt.conf, nms_thres=opt.nms)

        for i in range(len(outs)):
            path = paths[i]
            out = outs[i].numpy()
            for j in range(out.shape[0]):
                df_res = df_res.append({
                    'path': path,
                    'class': out[j, -1],
                    'x1': out[j, 0] / 416,
                    'y1': out[j, 1] / 416,
                    'x2': out[j, 2] / 416,
                    'y2': out[j, 3] / 416,
                    'confidence': out[j, -2]
                }, ignore_index=True)

    df_res.to_csv(opt.output, sep=',', index=None)
    print('-- yolo result stored at {}'.format(opt.output))

# defect detection pipeline
# author: Zhaonan Li, zli@brandeis.edu
# created at: 4/17/2020

import os
import argparse
import numpy as np
import pandas as pd
import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from PIL.ImageDraw import Draw
import time
from skimage.feature import peak_local_max
from grayscale_resnet import resnet18, resnet34, resnet50

from utils import *
from models import *

parser = argparse.ArgumentParser()
parser.add_argument("--hard", '-hd', type=str, required=True, help='path to weight of the hard scanner')
parser.add_argument("--uniform", '-u', type=str, required=True, help='path to weight of the uniform scanner')
parser.add_argument("--integrator", '-i', type=str, required=True, help='path to weight of integrator')
parser.add_argument("--images", type=str, required=True, help='path to image path file in .csv format')
parser.add_argument("--yolo", '-y', type=str, default=None, help='path to yolo result in .csv format')
parser.add_argument("--output", type=str, required=True, help='path to the output file')
parser.add_argument("--gpu", type=bool, default=True, help='use gpu or not')
parser.add_argument("--batch_size", '-b', type=int, default=128, help='batch size')

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
model_hard, model_uniform, model_int = load_models(opt.hard, opt.uniform, opt.integrator, device=device)
print('-- finished loading')

# if using gpu, wrap models with data parallel
if opt.gpu:
    model_hard = model_hard.cuda()
    model_uniform = model_uniform.cuda()
    model_int = model_int.cuda()

    model_hard = torch.nn.DataParallel(model_hard)
    model_uniform = torch.nn.DataParallel(model_uniform)
    model_int = torch.nn.DataParallel(model_int)

# load results from YOLO if applicable, select only positive and negative defects with confidence >= 0.5
# classes: 0 -> pos, 1 -> neg, 2 -> pos_o, 3 -> nuc (ignored)
df_yolo = None
if opt.yolo is not None:
    df_yolo = pd.read_csv(opt.yolo, sep=',')
    df_yolo = df_yolo[df_yolo['confidence'] >= 0.5]
    # df_yolo = df_yolo.drop(columns=['confidence', 'class'])

# load image paths
df_img_path = pd.read_csv(opt.images, sep=',')

# create result file
df_img_res = pd.DataFrame(columns=['path', 'class', 'x', 'y'])


if __name__ == '__main__':
    for idx, row in df_img_path.iterrows():
        path = row['path']

        print('\n-- evaluating {} ...'.format(path))
        since = time.time()

        img = Image.open(path).convert('L')
        w, h = img.size

        boxes = boxes_from_df(df_yolo, path)

        img_res = slide_on_image(img, model_hard, model_uniform, model_int, boxes, batch_size=opt.batch_size)
        res = np.transpose(img_res, (1, 2, 0))

        pos = peak_local_max(res[:, :, 0], min_distance=15, threshold_abs=0.983, threshold_rel=0.9).astype(np.float)
        pos_o = peak_local_max(res[:, :, 2], min_distance=12, threshold_abs=0.97, threshold_rel=0.9).astype(np.float)
        neg = peak_local_max(res[:, :, 1], min_distance=8, threshold_abs=0.988, threshold_rel=0.9).astype(np.float)

        if pos.size == 0:
            pos = np.array([]).reshape(0, 2)
        if pos_o.size == 0:
            pos_o = np.array([]).reshape(0, 2)
        if neg.size == 0:
            neg = np.array([]).reshape(0, 2)

        # merge pos and "pos open"
        pos = np.vstack((pos, pos_o))
        pos = np.array([[0, x / w, y / h] for x, y in pos])
        neg = np.array([[1, x / w, y / h] for x, y in neg])

        if pos.size == 0:
            pos = np.array([]).reshape(0, 3)
        if neg.size == 0:
            neg = np.array([]).reshape(0, 3)

        time_used = time.time() - since
        print('-- total {} positive defects and {} negative defects detected'.format(len(pos), len(neg)))
        print('-- finished {} in {:.0f}m {:.0f}s'.format(row['path'], time_used//60, time_used % 60))

        df_img = pd.DataFrame(np.vstack((pos, neg)), columns=['class', 'x', 'y'])
        df_img['path'] = path

        df_img_res = df_img_res.append(df_img, sort=True)

        df_img_res.to_csv(opt.output, sep=',', index=None)
        print('-- defect result stored at {}'.format(opt.output))

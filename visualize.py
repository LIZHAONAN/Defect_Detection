# helper methods for visualizing results
# author: Zhaonan Li, zli@brandeis.edu

import os
import pandas as pd
from utils import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--yolo", '-y', type=str, default=None, help='path to yolo result in .csv format')
parser.add_argument('--defects', '-d', type=str, default=None, help='path to defect result in .csv format')
parser.add_argument('--save', type=str, default='.', help='path to output directory')
opt = parser.parse_args()


def check_yolo_detection(df_yolo, save=None):
    for path in df_yolo['path'].unique().astype(str):
        boxes = boxes_from_df(df_yolo, path)
        img = visualize_yolo(path, boxes)
        if save is not None:
            file_name = os.path.join(save, 'yolo_' + os.path.basename(path))
            img.save(file_name)
            print('-- result stored at {}'.format(file_name))


def check_defect_detection(df_defects, save=None):
    for path in df_defects['path'].unique().astype(str):
        pts = pts_from_df(df_defects, path)
        img = visualize_pts(path, pts)
        if save is not None:
            file_name = os.path.join(save, 'defects_' + os.path.basename(path))
            img.save(file_name)
            print('-- result stored at {}'.format(file_name))


if __name__ == '__main__':
    if opt.yolo is not None:
        df_yolo = pd.read_csv(opt.yolo, sep=',')
        check_yolo_detection(df_yolo, save=opt.save)

    if opt.defects is not None:
        df_defects = pd.read_csv(opt.defects, sep=',')
        check_defect_detection(df_defects, save=opt.save)

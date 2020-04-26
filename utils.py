# author: Zhaonan Li, zli@brandeis.edu
# created at: 4/17/2020
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from PIL import Image
from PIL.ImageDraw import Draw

mean, std = 0.3019, 0.1909


# modified from create_circular_mask implemented by ruoshi
# create circular mask in the form of np array
def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = [int(w / 2), int(h / 2)]
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    mask = mask.astype(np.double)

    return mask


mask = create_circular_mask(200, 200)


# modified from sliding_window implemented by ruoshi
# slide over a given region of a image with given step size and window size
# the window is cropped, resized to 200 * 200, rescale to 0-1, masked, and normalized
# return batches of centers of windows (x and y) and windows
def sliding_window(image, stepSize, windowSize, xmin, xmax, ymin, ymax, mean=mean, std=std,
                   mask=mask, batch_size=128):
    if mean is None or std is None:
        mean, std = 0, 1

    # load x, y and input image
    pos = []
    data = []
    count = 0
    for x in range(xmin, xmax, stepSize):
        for y in range(ymin, ymax, stepSize):
            if image is None:
                pos.append([x, y])
                count += 1
            else:
                window = image.crop(box=(x - windowSize/2, y - windowSize/2, x + windowSize/2, y + windowSize/2))
                window = transforms.functional.resize(window, (200, 200))
                # normalize
                window = np.array(window) / np.max(np.array(window))
                # apply circular mask if given
                if mask is not None:
                    window = window * mask
                # normalize based on data set
                window = (window - mean) / std
                pos.append([x, y])
                data.append(window)
                count += 1
            if count == batch_size:
                yield pos, np.array(data).astype(float)
                count = 0
                pos = []
                data = []
    if len(pos) > 0:
        yield pos, np.array(data).astype(float)


# obtain results of two models
def detect_window(window, model_hard, model_uniform, model_int):
    model_hard.train(False)
    model_uniform.train(False)

    window = window.float()

    with torch.no_grad():
        out_hard = model_hard(window)  # [N, 5]
        out_uniform = model_uniform(window)  # [N, 5]
        out = torch.cat((out_uniform, out_hard), dim=1)
        out = model_int(out)
        out = F.softmax(out, dim=1)

        out = out.cpu().squeeze().numpy()
    return out


# run sliding window with both models on input image, with or without bounding boxes from YOLO
# the result is in size (10, W, H), [pos, neg, non] * 2: hard then uniform
def slide_on_image(img, model_hard, model_uniform, model_int, boxes, stride=2, batch_size=128):
    w, h = img.size

    img_res = np.vstack((np.zeros((4, w, h)), np.ones((1, w, h))))
    #     img_res = np.vstack((img_res, img_res))

    if boxes is not None:
        print('-- {} bounding boxes found, window slide on boxes'.format(len(boxes)))
        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            x1, x2 = [int(i * w) for i in [x1, x2]]
            y1, y2 = [int(i * h) for i in [y1, y2]]

            for pos, windows in sliding_window(img, stride, 45, x1, x2, y1, y2, batch_size=batch_size):

                windows = torch.from_numpy(windows).unsqueeze(1)

                res = detect_window(windows, model_hard, model_uniform, model_int)

                for i in range(len(pos)):
                    x, y = pos[i]
                    img_res[:, x, y] = res[i]
    else:
        print('-- no bounding boxes found, window slide whole image')
        x1, y1, x2, y2 = 0, 0, w, h
        for pos, windows in sliding_window(img, stride, 45, x1, x2, y1, y2, batch_size=batch_size):

            windows = torch.from_numpy(windows).unsqueeze(1)

            res = detect_window(windows, model_hard, model_uniform, model_int)

            for i in range(len(pos)):
                x, y = pos[i]
                img_res[:, x, y] = res[i]
    return img_res


# plot bounding boxes of YOLO on the input image
# input images can be in the form of Image instance, np array or path to a Image file
# return PIL Image with bounding boxes
def visualize_yolo(image_path, boxes):
    if isinstance(image_path, Image.Image):
        img = image_path
    elif isinstance(image_path, np.ndarray):
        img = Image.fromarray(image_path).convert('L')
    else:
        img = Image.open(image_path).convert('L')
    tmp = Image.new('RGB', img.size) # convert input image to rgb image
    tmp.paste(img)                    #
    img = tmp
    draw = Draw(img)
    w, h = img.size

    for box in boxes:
        x1, y1, x2, y2 = box
        #         y1, y2 = 1-y2, 1-y1 # optional
        x1, x2 = [int(i * w) for i in [x1, x2]]
        y1, y2 = [int(i * h) for i in [y1, y2]]

        draw.rectangle([x1, y1, x2, y2], outline='red')
    return img


# convert bounding boxes from dataframe to a list of 4 coordinates: xmin, ymin, xmax, ymax
def boxes_from_df(df, path):
    if df is None:
        return None
    df_img = df[df['path'] == path]
    # df_img = df_img.drop(columns=['path', 'confidence', 'class'])
    df_img = df_img[['x1', 'y1', 'x2', 'y2']]
    if df_img.empty:
        return None
    return df_img.values


# plot points on the input image
# input images can be in the form of Image instance, np array or path to a Image file
# return PIL Image with points
def visualize_pts(image_path, pts):
    if isinstance(image_path, Image.Image):
        img = image_path.copy()
    elif isinstance(image_path, np.ndarray):
        img = Image.fromarray(image_path).convert('L')
    else:
        img = Image.open(image_path).convert('L')
    tmp = Image.new('RGB', img.size)  # convert input image to rgb image
    tmp.paste(img)  #
    img = tmp
    draw = Draw(img)
    w, h = img.size

    for pt in pts:
        c, x, y = pt

        x1, x2 = int(max(x * w - 5, 0)), int(min(x * w + 5, w))
        y1, y2 = int(max(y * h - 5, 0)), int(min(y * h + 5, h))
        color = ['blue', 'red'][int(c)]
        draw.rectangle([x1, y1, x2, y2], outline=color)

    return img


# convert defect positions from dataframe to a list of 3 values: class, x, y
def pts_from_df(df, path):
    df_img = df[df['path'] == path]
    # df_img = df_img.drop(columns=['path'])
    # df_img['y'] = 1 - df_img['y']
    df_img = df_img[['class', 'x', 'y']]
    return df_img.values

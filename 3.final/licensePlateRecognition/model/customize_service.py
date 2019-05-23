from PIL import Image
from model_service.pytorch_model_service import PTServingBaseService

import torch.nn as nn
import torch.nn.functional as F
import torch
import json
import os
import numpy as np
import cv2
from models import *

import torchvision.transforms as transforms

infer_transformation = transforms.Compose([
    transforms.Resize((32, 32), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# 预处理后处理类的父类需要定义为PTServingBaseService，
class PTVisionService(PTServingBaseService):

    def __init__(self, model_name, model_path):

        super(PTVisionService, self).__init__(model_name, model_path)

        # self.model定义为用户load后的模型
        # 当前ModelArts平台PyTorch只支持cpu推理，因此需要映射设备到cpu
        # 详细说明可以参考https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices
        device = torch.device('cpu')

        # 模型网络结构
        img_size = 416
        cfg = '/home/mind/model/yolov3-spp.cfg'
        self.model = Darknet(cfg, img_size)
        # load模型变量
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()


    def _preprocess(self, data):

        # 预处理成{key:input_batch_var}，input_batch_var为模型输入张量
        preprocessed_data = {}
        for k, v in data.items():
            input_batch = []
            for file_name, file_content in v.items():
                with Image.open(file_content) as image1:
                    img0 = image1.convert("RGB")
                    img0 = np.array(img0)
                    img, _, _, _ = letterbox(img0, height=416)
                    img = img.transpose(2, 0, 1)
                    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    img = torch.from_numpy(img)
                    input_batch.append(img)
            input_batch_var = torch.autograd.Variable(torch.stack(input_batch, dim=0), volatile=True)
            preprocessed_data[k] = input_batch_var

        return preprocessed_data

    def _postprocess(self, data):
        # data输出为{key:output_batch_var},output_batch_var为模型输出的张量

        label2char = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                      10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'J', 19: 'K',
                      20: 'L', 21: 'M', 22: 'N', 23: 'P', 24: 'Q', 25: 'R', 26: 'S', 27: 'T', 28: 'U', 29: 'V',
                      30: 'W', 31: 'X', 32: 'Y', 33: 'Z',
                      34: '0', 35: '1', 36: '2', 37: '3', 38: '4', 39: '5', 40: '6', 41: '7', 42: '8'}

        result = {}
        # 根据标签索引到图片的分类结果
        for k, v in data.items():

            pred = v
            conf_thres = 0.5
            nms_thres = 0.5
            det = non_max_suppression(pred, conf_thres, nms_thres)[0]

            det[:, :4] = scale_coords((96, 416), det[:, :4], (70, 360, 3)).round()

            if det.shape[0] > 9:
                det = det[0:9, :]
            _, sort_idx = torch.sort(det[:, 0])
            det = det[sort_idx]

            if det.shape[0] < 9:
                conf_thres1 = conf_thres
                while det.shape[0] < 9:
                    conf_thres1 = conf_thres1 - 0.1
                    det = non_max_suppression(pred, conf_thres1, nms_thres)[0]
                    if conf_thres1 < 0.1:
                        break

            det_numpy = det.cpu().detach().numpy()

            if det_numpy.shape[0] > 9:
                l = det_numpy.shape[0] - 9
                for k in range(l):
                    d = det_numpy[:, 0]
                    dc = np.zeros(len(d) - 1)
                    for j in range(len(dc)):
                        dc[j] = d[j + 1] - d[j]
                    row1 = np.argsort(dc)[0]
                    row2 = row1 + 1
                    conf1 = det_numpy[row1, 4]
                    conf2 = det_numpy[row2, 4]

                    if conf1 > conf2:
                        det_numpy = np.delete(det_numpy, row2, 0)
                    else:
                        det_numpy = np.delete(det_numpy, row1, 0)

            l_char = det_numpy[:, 6]
            result = ''
            for j in range(len(l_char)):
                result += label2char[int(l_char[j])]

        # 输出result为{key: 图片类别}
        return result


def letterbox(img, height=416, color=(127.5, 127.5, 127.5), mode='rect'):
    # Resize a rectangular image to a 32 pixel multiple rectangle
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]

    # Select padding https://github.com/ultralytics/yolov3/issues/232
    if mode is 'rect':  # rectangle
        dw = np.mod(height - new_shape[0], 32) / 2  # width padding
        dh = np.mod(height - new_shape[1], 32) / 2  # height padding
    else:  # square
        dw = (height - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratio, dw, dh


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.5):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_conf, class)
    """

    min_wh = 2  # (pixels) minimum box width and height

    output = [None] * len(prediction)
    for image_i, pred in enumerate(prediction):

        class_conf, class_pred = pred[:, 5:].max(1)
        pred[:, 4] *= class_conf

        # Select only suitable predictions
        i = (pred[:, 4] > conf_thres) & (pred[:, 2:4] > min_wh).all(1) & (torch.isnan(pred).any(1) == 0)
        pred = pred[i]

        # If none are remaining => process next image
        if len(pred) == 0:
            continue

        # Select predicted classes
        class_conf = class_conf[i]
        class_pred = class_pred[i].unsqueeze(1).float()

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])
        # pred[:, 4] *= class_conf  # improves mAP from 0.549 to 0.551

        # Detections ordered as (x1y1x2y2, obj_conf, class_conf, class_pred)
        pred = torch.cat((pred[:, :5], class_conf.unsqueeze(1), class_pred), 1)

        # Get detections sorted by decreasing confidence scores
        # pred = pred[(-pred[:, 4]).argsort()]

        _, sort_idx = torch.sort(-pred[:, 4])
        pred = pred[sort_idx]

        det_max = []
        nms_style = 'MERGE'  # 'OR' (default), 'AND', 'MERGE' (experimental)
        for c in pred[:, -1].unique():
            dc = pred[pred[:, -1] == c]  # select class c
            dc = dc[:min(len(dc), 100)]  # limit to first 100 boxes: https://github.com/ultralytics/yolov3/issues/117

            # No NMS required if only 1 prediction
            if len(dc) == 1:
                det_max.append(dc)
                continue

            # Non-maximum suppression
            if nms_style == 'OR':  # default
                # METHOD1
                # ind = list(range(len(dc)))
                # while len(ind):
                # j = ind[0]
                # det_max.append(dc[j:j + 1])  # save highest conf detection
                # reject = (bbox_iou(dc[j], dc[ind]) > nms_thres).nonzero()
                # [ind.pop(i) for i in reversed(reject)]

                # METHOD2
                while dc.shape[0]:
                    det_max.append(dc[:1])  # save highest conf detection
                    if len(dc) == 1:  # Stop if we're at the last detection
                        break
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'AND':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'MERGE':  # weighted mixture box
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    i = bbox_iou(dc[0], dc) > nms_thres  # iou with other boxes
                    weights = dc[i, 4:5]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[i == 0]

        if len(det_max):
            det_max = torch.cat(det_max)  # concatenate

            # output[image_i] = det_max[(-det_max[:, 4]).argsort()]  # sort

            _, sort_idx = torch.sort(-det_max[:, 4])
            output[image_i] = det_max[sort_idx]

    return output


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def bbox_iou(box1, box2, x1y1x2y2=True):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    return inter_area / union_area  # iou


def scale_coords(img1_shape, coords, img0_shape):
    # Rescale coords1 (xyxy) from img1_shape to img0_shape
    gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
    coords[:, [0, 2]] -= (img1_shape[1] - img0_shape[1] * gain) / 2  # x padding
    coords[:, [1, 3]] -= (img1_shape[0] - img0_shape[0] * gain) / 2  # y padding
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(min=0)
    return coords
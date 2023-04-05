"""This module contains simple helper functions """
from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import torch.nn.functional as F


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


class MeanAbsoluteError(object):
    def __init__(self):
        self.mae_list = []

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        batch_size, c, h, w = gt.shape
        assert batch_size == 1, f"validation mode batch_size must be 1, but got batch_size: {batch_size}."
        resize_pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=False)
        error_pixels = torch.sum(torch.abs(resize_pred - gt), dim=(1, 2, 3)) / (h * w)
        self.mae_list.extend(error_pixels.tolist())

    def compute(self):
        mae = sum(self.mae_list) / len(self.mae_list)
        return mae

    def __str__(self):
        mae = self.compute()
        return f'MAE: {mae:.5f}'


class F1Score(object):
    """
    refer: https://github.com/xuebinqin/DIS/blob/main/IS-Net/basics.py
    """

    def __init__(self, threshold: float = 0.0):
        # 初始化四个计数器
        self.tp = 0  # 真正例
        self.fp = 0  # 假正例
        self.fn = 0  # 假反例
        self.tn = 0  # 真反例

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        assert pred.shape == gt.shape
        # 创建新的张量来转换为二值张量
        pred_clone = pred.clone()
        gt_clone = gt.clone()
        self.tp += torch.sum((pred_clone < 0) & (gt_clone < 0)).item()
        self.fp += torch.sum((pred_clone < 0) & (gt_clone >= 0)).item()
        self.fn += torch.sum((pred_clone >= 0) & (gt_clone < 0)).item()
        self.tn += torch.sum((pred_clone >= 0) & (gt_clone >= 0)).item()

    def compute(self):
        # 计算f1-score，注意分母可能为零的情况
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        return f1_score

    def __str__(self):
        max_f1 = self.compute()
        return f'maxF1: {max_f1:.5f}'


# 计算两个torch.Tensor（pred和gt）的mIoU，tensor的shape是batch_size, c, h, w，其中batch_size为1，c为1，每个像素的值在-1到1之间，图像仅包含前景和背景，背景像素均为白色。
def compute_miou(pred, gt):
    # Flatten the tensors and convert them to 1D arrays
    pred = pred.view(-1)
    gt = gt.view(-1)

    # Calculate the intersection and union of the predicted and ground truth masks
    intersection = torch.sum((pred < 0) & (gt < 0)).float()
    union = torch.sum((pred < 0) | (gt < 0)).float()

    # Calculate the IoU and return it
    iou = intersection / union
    return iou.item()


class MeanIoU():
    def __init__(self):
        self.miou_list = []

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        batch_size, c, h, w = gt.shape
        assert batch_size == 1, f"validation mode batch_size must be 1, but got batch_size: {batch_size}."
        # resize_pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=False)
        # resize_pred = torch.argmax(resize_pred, dim=1)
        # gt = torch.argmax(gt, dim=1)
        miou = compute_miou(pred, gt)
        self.miou_list.append(miou)

    def compute(self):
        miou = sum(self.miou_list) / len(self.miou_list)
        return miou

    def __str__(self):
        miou = self.compute()
        return f'MIOU: {miou:.6f}'


class Acc:
    def __init__(self):
        self.acc_list = []

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        batch_size, c, h, w = gt.shape
        assert batch_size == 1, f"validation mode batch_size must be 1, but got batch_size: {batch_size}."
        resize_pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=False)
        resize_pred = torch.argmax(resize_pred, dim=1)
        gt = torch.argmax(gt, dim=1)
        acc = torch.sum(torch.eq(resize_pred, gt).float()) / (h * w)
        self.acc_list.append(acc.item())

    def compute(self):
        acc = sum(self.acc_list) / len(self.acc_list)
        return acc

    def __str__(self):
        acc = self.compute()
        return f'Acc: {acc:.5f}'


# 计算敏感性
class Sensitivity:
    def __init__(self):
        # 初始化两个计数器
        self.tp = 0  # 真正例
        self.fn = 0  # 假反例

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        # 更新计数器，假设pred和gt都是形状为（1，1，512，512）的张量
        assert pred.shape == gt.shape
        # 创建新的张量来转换为二值张量
        pred_clone = pred.clone()
        gt_clone = gt.clone()
        self.tp += torch.sum((pred_clone < 0) & (gt_clone < 0)).item()
        self.fn += torch.sum((pred_clone < 0) & (gt_clone >= 0)).item()

    def compute(self):
        # 计算敏感度Sen，注意分母可能为零的情况
        sen = self.tp / (self.tp + self.fn) if self.tp + self.fn > 0 else 0.0
        return sen

    def __str__(self):
        sen = self.compute()
        return f'Sensitivity: {sen:.5f}, TP: {self.tp}, FN: {self.fn}'


class RMSE:
    def __init__(self):
        # 初始化一个MSELoss对象
        self.rmse_loss = None
        self.mse_loss = []
        self.mse = nn.MSELoss()

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        # 计算MSE
        self.mse_loss.append(self.mse(pred, gt))

    def compute(self):
        mean_mes = sum(self.mse_loss) / len(self.mse_loss)
        # 计算RMSE，注意开根号可能出现NaN的情况
        self.rmse_loss = torch.sqrt(mean_mes) if mean_mes > 0 else 0.0
        return self.rmse_loss

    def __str__(self):
        rmse = self.compute()
        return f'RMSE: {rmse:.5f}'


class Specificity:
    def __init__(self):
        # 初始化两个计数器
        self.tn = 0 # 真负例
        self.fp = 0 # 假正例

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        # 更新计数器，假设pred和gt都是形状为（1，1，512，512）的张量
        assert pred.shape == gt.shape
        # 创建新的张量来转换为二值张量
        pred_bin = pred.clone()
        gt_bin = gt.clone()
        self.tn += torch.sum((pred_bin >= 0) & (gt_bin >= 0)).item()
        self.fp += torch.sum((pred_bin < 0) & (gt_bin >= 0)).item()

    def compute(self):
        # 计算特异性Specificity，注意分母可能为零的情况
        specificity = self.tn / (self.tn + self.fp) if self.tn + self.fp > 0 else 0.0
        return specificity

    def __str__(self):
        specificity = self.compute()
        return f'Specificity: {specificity:.5f}, TN: {self.tn}, FP: {self.fp}'
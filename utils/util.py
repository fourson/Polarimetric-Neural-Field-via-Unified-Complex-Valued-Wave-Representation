import os

import torch
import cv2
import numpy as np

from model.metric_utils.psnr import PSNR
from model.metric_utils.ssim import SSIM


def read_img(path, rgb=True):
    img = cv2.imread(str(path), -1)
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(img) / 255.
    return img


def write_img(path, img, rgb=True):
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img * 255
    cv2.imwrite(str(path), img)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def compute_metrics_for_tensor(pred, gt, input_type, avg_channel_pol=False):
    with torch.no_grad():
        assert input_type in ('S0', 'p', 'theta'), "input_type must be S0, p, or theta"
        psnr = PSNR().cuda()
        ssim = SSIM(channel=3).cuda()
        md = dict()

        # normalize
        if input_type == 'S0':
            pred, gt = pred / 2, gt / 2
        elif input_type == 'theta':
            pred, gt = pred / np.pi, gt / np.pi
        else:
            pass

        if input_type in ('p', 'theta') and avg_channel_pol:
            pred_avg = torch.mean(pred, dim=1, keepdim=True)
            pred = torch.cat([pred_avg] * 3, dim=1)
            gt_avg = torch.mean(gt, dim=1, keepdim=True)
            gt = torch.cat([gt_avg] * 3, dim=1)

        md[f'psnr_{input_type}'] = psnr(pred, gt).item()
        md[f'ssim_{input_type}'] = ssim(pred, gt).item()

        return md


def visualize_array(path, x, input_type, avg_channel_pol=False):
    assert input_type in ('S0', 'p', 'theta'), "input_type must be S0, p, or theta"

    # normalize
    if input_type == 'S0':
        x = x / 2
    elif input_type == 'theta':
        x = x / np.pi
    else:
        pass

    if input_type in ('p', 'theta') and avg_channel_pol:
        # if avg_channel_pol, we visualize p and theta using color map
        color_map = cv2.applyColorMap(cv2.cvtColor(np.uint8(x * 255), cv2.COLOR_RGB2GRAY), cv2.COLORMAP_JET)
        cv2.imwrite(path, color_map)
    else:
        write_img(path, x, rgb=True)


def visualize_error_map_for_array(path, pred, gt, input_type):
    assert input_type in ('S0', 'p', 'theta'), "input_type must be S0, p, or theta"

    # normalize
    if input_type == 'S0':
        pred, gt = pred / 2, gt / 2
    elif input_type == 'theta':
        pred, gt = pred / np.pi, gt / np.pi
    else:
        pass

    error = np.mean(np.abs(pred - gt), axis=2)
    error_map = cv2.applyColorMap(np.uint8(error * 255), cv2.COLORMAP_JET)
    write_img(path, error_map, rgb=True)


def get_lr_lambda(lr_lambda_tag):
    if lr_lambda_tag == 'default':
        # keep the same
        return lambda epoch: 1
    elif lr_lambda_tag == '2000_3quarter':
        return lambda epoch: 0.75 ** (epoch // 2000)
    elif lr_lambda_tag == '2000_half':
        return lambda epoch: 0.5 ** (epoch // 2000)
    elif lr_lambda_tag == '2000_quarter':
        return lambda epoch: 0.25 ** (epoch // 2000)
    elif lr_lambda_tag == '2000_deci':
        return lambda epoch: 0.1 ** (epoch // 2000)
    else:
        raise NotImplementedError('lr_lambda_tag [%s] is not found' % lr_lambda_tag)

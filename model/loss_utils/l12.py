import torch
import torch.nn.functional as F


def l1(pred, gt):
    return F.l1_loss(pred, gt)


def l2(pred, gt):
    return F.mse_loss(pred, gt)


def complex_l1(pred, gt):
    return l1(pred.real, gt.real) + l1(pred.imag, gt.imag)


def complex_l2(pred, gt):
    return l2(pred.real, gt.real) + l2(pred.imag, gt.imag)

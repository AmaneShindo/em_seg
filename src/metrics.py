import torch

def _th(pred, thr=0.5):
    return (torch.sigmoid(pred) > thr).float()

def dice(pred, target, eps=1e-6):
    pred = _th(pred)
    inter = (pred * target).sum()
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)

def iou(pred, target, eps=1e-6):
    pred = _th(pred)
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + eps) / (union + eps)

def precision(pred, target, eps=1e-6):
    pred = _th(pred)
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    return (tp + eps) / (tp + fp + eps)

def recall(pred, target, eps=1e-6):
    pred = _th(pred)
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return (tp + eps) / (tp + fn + eps)

def f1(pred, target, eps=1e-6):
    p = precision(pred, target, eps)
    r = recall(pred, target, eps)
    return 2 * p * r / (p + r + eps)

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):  # 这个用了
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        box1 = torch.cat((box1[..., :2] - box1[..., 2:] / 2, box1[..., :2] + box1[..., 2:] / 2), dim=-1)
        box2 = torch.cat((box2[..., :2] - box2[..., 2:] / 2, box2[..., :2] + box2[..., 2:] / 2), dim=-1)

    inter = torch.max(box1[:, None, :2], box2[:, :2]) - torch.min(box1[:, None, 2:], box2[:, 2:])
    inter = torch.clamp(inter, min=0)
    inter_area = inter[..., 0] * inter[..., 1]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = area1[:, None] + area2 - inter_area

    return inter_area / union_area

def bbox_giou(box1, box2, x1y1x2y2=True):
    iou = bbox_iou(box1, box2, x1y1x2y2)

    if not x1y1x2y2:
        box1 = torch.cat((box1[..., :2] - box1[..., 2:] / 2, box1[..., :2] + box1[..., 2:] / 2), dim=-1)
        box2 = torch.cat((box2[..., :2] - box2[..., 2:] / 2, box2[..., :2] + box2[..., 2:] / 2), dim=-1)

    c = torch.max(box1[:, None, 2:], box2[:, 2:]) - torch.min(box1[:, None, :2], box2[:, :2])
    c_area = c[..., 0] * c[..., 1]

    return iou - (c_area - iou) / c_area

def bbox_diou(box1, box2, x1y1x2y2=True):
    iou = bbox_iou(box1, box2, x1y1x2y2)

    if not x1y1x2y2:
        box1 = torch.cat((box1[..., :2] - box1[..., 2:] / 2, box1[..., :2] + box1[..., 2:] / 2), dim=-1)
        box2 = torch.cat((box2[..., :2] - box2[..., 2:] / 2, box2[..., :2] + box2[..., 2:] / 2), dim=-1)

    center1 = (box1[:, :2] + box1[:, 2:]) / 2
    center2 = (box2[:, :2] + box2[:, 2:]) / 2
    inter_diag = torch.sum((center2 - center1) ** 2, dim=-1)

    c = torch.max(box1[:, None, 2:], box2[:, 2:]) - torch.min(box1[:, None, :2], box2[:, :2])
    c_diag = torch.sum(c ** 2, dim=-1)

    return iou - inter_diag / c_diag


def bbox_ciou(box1, box2, x1y1x2y2=True):
    # 首先计算DIoU
    diou = bbox_diou(box1, box2, x1y1x2y2)
    
    if x1y1x2y2:
        # 转换为[Cx, Cy, W, H]格式
        box1 = torch.cat(((box1[..., 2:] + box1[..., :2]) / 2, box1[..., 2:] - box1[..., :2]), dim=-1)
        box2 = torch.cat(((box2[..., 2:] + box2[..., :2]) / 2, box2[..., 2:] - box2[..., :2]), dim=-1)
    
    w1, h1 = box1[:, 2], box1[:, 3]
    w2, h2 = box2[:, 2], box2[:, 3]
    
    # 计算长宽比的一致性
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w1 / h1) - torch.atan(w2 / h2), 2)
    
    # 计算alpha参数，以避免当重叠区域为0时长宽比项过高的惩罚
    with torch.no_grad():
        alpha = v / (1 - diou + v)
    
    # 最终的CIoU值包括DIoU值和长宽比的一致性
    ciou = diou - (alpha * v)
    return ciou
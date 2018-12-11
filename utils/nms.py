import torch

#from nms._ext import nms
import torch
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def box_nms(bboxes, threshold=0.5, mode='union'):
    '''Non maximum suppression.
    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) bbox scores, sized [N,].
      threshold: (float) overlap threshold.
      mode: (str) 'union' or 'min'.
    Returns:
      keep: (tensor) selected indices.
    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    areas = (x2-x1+1) * (y2-y1+1)
    #_, order = scores.sort(0, descending=True)
    order = torch.LongTensor(list(range(0, len(bboxes))))
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=int(x1[i]))
        yy1 = y1[order[1:]].clamp(min=int(y1[i]))
        xx2 = x2[order[1:]].clamp(max=int(x2[i]))
        yy2 = y2[order[1:]].clamp(max=int(y2[i]))

        w = (xx2-xx1+1).clamp(min=0)
        h = (yy2-yy1+1).clamp(min=0)
        inter = w*h

        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / areas[order[1:]].clamp(max=areas[i])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)

        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)

class NMS(object):
    
    @staticmethod
    def suppress(sorted_bboxes , threshold):
        keep_indices = torch.tensor([], dtype=torch.long).to(device)
        #nms.suppress(sorted_bboxes.contiguous(), threshold, keep_indices)
        #print (sorted_bboxes)
        #sorted_bboxes = sorted_bboxes.detach().numpy()
        keep_indices = box_nms(sorted_bboxes, threshold= threshold)
        keep_indices = torch.LongTensor(keep_indices)
        return keep_indices

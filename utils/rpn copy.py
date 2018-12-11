from typing import Tuple
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn, Tensor
from utils.bbox import BBox
from utils.nms import NMS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RegionProposalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self._features = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self._objectness = nn.Conv2d(in_channels = 512, out_channels= 18, kernel_size= 1)
        self._transformer = nn.Conv2d(in_channels= 512, out_channels= 36, kernel_size=1)

    def forward(self, features, image_width, image_height):
        anchor_boxes = RegionProposalNetwork._generate_anchors(image_width, image_height, num_x_anchors = features.shape[3], num_y_anchors = features.shape[2]).to(DEVICE)
        features = self._features(features)
        objectnesses = self._objectness(features)
        transformer = self._transformer(features)

        objectnesses = objectnesses.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        transformer = transformer.permute(0,2, 3,1).contiguous().view(-1, 4)

        proposal_boxes = RegionProposalNetwork._generate_proposals(anchor_boxes, objectnesses, transformer, image_width, image_height)
        proposal_boxes = proposal_boxes[:12000 if self.training else 6000]
        keep_indices = NMS.suppress(proposal_boxes, threshold = 0.7)
        proposal_boxes = proposal_boxes[keep_indices]
        proposal_boxes = proposal_boxes[:2000 if self.training else 300]
        return anchor_boxes, objectnesses, transformer, proposal_boxes

    def sample(self, anchor_boxes, anchor_objectnesses, anchor_transformers, 
                gt_boxes, image_width, image_height):
        anchor_boxes = anchor_transformers.cpu()
        gt_boxes = gt_boxes.cpu()
        boundary = torch.tensor(BBox(0, 0, image_width, image_height).tolist(), dtype=torch.float)
        print (anchor_boxes.shape)
        inside_indices = BBox.inside(anchor_boxes, boundary.unsqueeze(dim = 0)).squeeze().nonzero().view(-1)
        print (len(inside_indices))
        anchor_boxes = anchor_boxes[inside_indices]
        anchor_objectnesses = anchor_objectnesses[inside_indices]
        anchor_transformers = anchor_transformers[inside_indices]
        
        #finding labels for each anchor
        labels = torch.ones(len(anchor_boxes), dtype = torch.long) * -1
        #print ('**',anchor_boxes.shape)
        ious = BBox.iou(anchor_boxes, gt_boxes)
        anchor_max_ious, anchor_assignments = ious.max(dim = 1)
        gt_max_ious, gt_assignments = ious.max(dim = 0)
        anchor_additions = (ious == gt_max_ious).nonzero()[:, 0]
        labels[anchor_max_ious < 0.3] = 0
        labels[anchor_additions] = 1
        labels[anchor_max_ious >= 0.7] = 1

        #select 256 samples
        fg_indices = (labels == 1).nonzero().view(-1)
        bg_indices = (labels == 1).nonzero().view(-1)
        fg_indices = fg_indices[torch.randperm(len(fg_indices))[:min(len(fg_indices), 128)]]
        bg_indices = bg_indices[torch.randperm(len(bg_indices))[:256 - len(fg_indices)]]
        select_indices = torch.cat([fg_indices, bg_indices])

        gt_anchor_objectnesses = labels[select_indices]
        gt_boxes = gt_boxes[anchor_assignments[fg_indices]]
        anchor_boxes = anchor_boxes[fg_indices]
        gt_anchor_transformers = BBox.calc_transformer(anchor_boxes, gt_boxes)

        gt_anchor_objectnesses = gt_anchor_objectnesses.to(DEVICE)
        gt_anchor_transformers = gt_anchor_transformers.to(DEVICE)

        anchor_objectnesses = anchor_objectnesses[select_indices]
        anchor_transformers = anchor_transformers[fg_indices]

        return anchor_objectnesses, anchor_transformers, gt_anchor_objectnesses, gt_anchor_transformers
    
    def loss(self, anchor_objectnesses, anchor_transformers, gt_anchor_objectnesses, gt_anchor_transformer):
        cross_entropy = F.cross_entropy(input = anchor_objectnesses, target= gt_anchor_objectnesses)
        gt_anchor_transformer = gt_anchor_transformer.detach()
        smooth_l1_loss = F.smooth_l1_loss(input = anchor_transformers, target = gt_anchor_transformer, reduction = 'sum')
        smooth_l1_loss /= len(gt_anchor_transformer)
        return cross_entropy, smooth_l1_loss
    
    @staticmethod
    def _generate_anchors(image_width, image_height, num_x_anchors, num_y_anchors):
        center_based_anchor_boxes = []
        for anchor_y in np.linspace(start = 0, stop=image_height, num=num_y_anchors + 2)[1:-1]:
            for anchor_x in np.linspace(start = 0, stop = image_width, num = num_x_anchors +2)[1: -1]:
                for ratio in [(1,2),(1,1),(2,1)]:
                    for size  in [128, 256, 512]:
                        center_x  = float(anchor_x)
                        center_y = float(anchor_y)
                        r = ratio[0] / ratio[1]
                        height = size * np.sqrt(r)
                        width = size * np.sqrt(1/r)
                        center_based_anchor_boxes.append([center_x, center_y, width, height])
        center_based_anchor_boxes = torch.tensor(center_based_anchor_boxes, dtype = torch.float)
        anchor_boxes = BBox.from_center_base(center_based_anchor_boxes)
        return anchor_boxes
    @staticmethod
    def _generate_proposals(anchor_bboxes, objectnesses, transformers, image_width, image_height):
        proposal_score = objectnesses[:,1]
        _, sorted_indices = torch.sort(proposal_score, dim = 0, descending = True)
        #anchor_bboxes = anchor_bboxes.permute(1,0)
        sorted_transformers = transformers[sorted_indices]
        sorted_anchor_boxes = anchor_bboxes[sorted_indices]
        proposal_boxes = BBox.apply_transformer(sorted_anchor_boxes, sorted_transformers.detach())
        proposal_boxes = BBox.clip(proposal_boxes, 0, 0, image_width, image_height)
        area_threshold = 16
        non_small_area_indices = ((proposal_boxes[:, 2] - proposal_boxes[:, 0] >= area_threshold) & 
                                (proposal_boxes[:, 3] - proposal_boxes[:, 1] >= area_threshold)).nonzero().view(-1)
        proposal_boxes = proposal_boxes[non_small_area_indices]
        return proposal_boxes
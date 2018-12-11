import os
import time
from typing import Union, Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from utils.base_model import BaseModel
from utils.bbox import BBox
from utils.nms import NMS
from utils.rpn import RegionProposalNetwork

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Model(nn.Module):
    NUM_CLASSES = 2

    class ForwardInput:
        class Train(object):
            def __init__(self, image, act_classes, act_bboxes):
                self.image = image
                self.act_classes = act_classes
                self.act_bboxes = act_bboxes
        class Eval(object):
            def __init__(self, image):
                self.image = image

    class ForwardOutput:
        class Train(object):
            def __init__(self, anchor_class_loss, anchor_transformer_loss, proposal_class_loss, proposal_transformer_loss):
                self.anchor_class_loss = anchor_class_loss
                self.anchor_transformer_loss = anchor_transformer_loss
                self.proposal_class_loss = proposal_class_loss
                self.proposal_transformer_loss = proposal_transformer_loss
        class Eval(object):
            def __init__(self, detection_labels, detection_bboxes, detection_probs):
                self.detection_labels = detection_labels
                self.detection_bboxes = detection_bboxes
                self.detection_probs = detection_probs
    
    def __init__(self, basemodel):
        super().__init__()
        self.features = basemodel.features()
        self.bn_modules = [module for module in self.features.modules()
                                if isinstance(module, nn.BatchNorm2d)]
        self.rpn = RegionProposalNetwork()
        self.detection = Model.Detection()

        self._transformer_normalize_mean = torch.tensor([0., 0., 0., 0.], dtype=torch.float)
        self._transformer_normalize_std = torch.tensor([.1, .1, .2, .2], dtype=torch.float)
    def forward(self, forward_input):
        for bn_module in self.bn_modules:
            bn_module.eval()
            for parameter in bn_module.parameters():
                parameter.requires_grad = False
        image = forward_input.image.unsqueeze(dim =  0)
        image_height, image_width = image.shape[2], image.shape[3]
        features = self.features(image)
        anchor_bboxes, anchor_classes_score, anchor_transformers, proposal_bboxes = self.rpn.forward(features, image_width, image_height)

        if self.training:
            anchor_classes_score, anchor_transformers, act_anchor_classes_score, act_anchor_transformers = self.rpn.sample(anchor_bboxes, anchor_classes_score, anchor_transformers, forward_input.act_bboxes, image_width, image_height)
            #print (anchor_transformers.shape)
            #print (act_anchor_transformers.shape)
            anchor_class_loss, anchor_transformer_loss = self.rpn.loss(anchor_classes_score, anchor_transformers, act_anchor_classes_score, act_anchor_transformers)
            proposal_bboxes, act_proposal_classes, act_proposal_transformer = self.sample(proposal_bboxes, forward_input.act_classes, forward_input.act_bboxes)
            proposal_classes, proposal_transformers = self.detection.forward(features, proposal_bboxes)
            proposal_class_loss, proposal_transformer_loss = self.loss(proposal_classes, proposal_transformers, act_proposal_classes, act_proposal_transformer) 
            forward_ouput = Model.ForwardOutput.Train(anchor_class_loss, anchor_transformer_loss, proposal_class_loss, proposal_transformer_loss)
        else:
            proposal_classes, proposal_transformers = self.detection.forward(features, proposal_bboxes)
            detection_bboxes, detection_labels, detection_probs = self._generate_detections(proposal_bboxes, proposal_classes, proposal_transformers, image_width, image_height)
            forward_ouput = Model.ForwardOutput.Eval(detection_labels, detection_bboxes, detection_probs)
        return forward_ouput
    
    def sample(self, proposal_bboxes, act_classes, act_bboxes):
        proposal_bboxes = proposal_bboxes.to(DEVICE)
        act_classes = act_classes.to(DEVICE)
        act_bboxes = act_bboxes.to(DEVICE)
        labels = torch.ones(len(proposal_bboxes), dtype = torch.long) * -1
        ious = BBox.iou(proposal_bboxes, act_bboxes)
        proposal_max_ious, proposal_assignments = ious.max(dim = 1)
        labels[proposal_max_ious < 0.5] = 0 
        labels[proposal_max_ious >= 0.5] = act_classes[proposal_assignments[proposal_max_ious >= 0.5]]
        #Yet to refine
        foreground_indices = (labels > 0).nonzero().view(-1)
        background_indices = (labels == 0).nonzero().view(-1)
        foreground_indices = foreground_indices[torch.randperm(len(foreground_indices))[:min(len(foreground_indices), 32)]]
        background_indices = background_indices[torch.randperm(len(background_indices))[:128 - len(foreground_indices)]]
        select_indices = torch.cat([foreground_indices, background_indices])
        select_indices = select_indices[torch.randperm(len(select_indices))]

        proposal_bboxes = proposal_bboxes[select_indices]
        act_proposal_transformers = BBox.calc_transformer(proposal_bboxes, act_bboxes[proposal_assignments[select_indices]])
        act_proposal_classes = labels[select_indices]
        act_proposal_transformers = (act_proposal_transformers - self._transformer_normalize_mean) / self._transformer_normalize_std
        act_proposal_classes = act_proposal_classes.to(DEVICE)
        return proposal_bboxes, act_proposal_classes, act_proposal_transformers

    def loss(self, proposal_classes, proposal_transformers, act_proposal_classes, act_proposal_transformers):
        cross_entropy = F.cross_entropy(input = proposal_classes, target = act_proposal_classes)
        proposal_transformers = proposal_transformers.view(-1, Model.NUM_CLASSES, 4)
        proposal_transformers = proposal_transformers[torch.arange(end = len(proposal_transformers), dtype = torch.long).to(DEVICE), act_proposal_classes]
        
        foreground_indices = act_proposal_classes.nonzero().view(-1)
        smooth_l1_loss = F.smooth_l1_loss(input = proposal_transformers[foreground_indices], target = act_proposal_transformers[foreground_indices], reduction = 'sum')
        smooth_l1_loss /= len(act_proposal_transformers)
        return cross_entropy, smooth_l1_loss
    
    def save(self, path_to_checkpoints_dir, step):
        path_to_checkpoint = os.path.join(path_to_checkpoints_dir, 
                                            'model-{}-{}.pth'.format(time.strftime('%y%m%d%H%M'), step))
        torch.save(self.state_dict(), path_to_checkpoint)
        return path_to_checkpoint
    
    def load(self, path_to_checkpoint):
        self.load_state_dict(torch.load(path_to_checkpoint, map_location= 'cpu'))
        return self
    
    def _generate_detections(self, proposal_bboxes, proposal_classes, proposal_transformers, image_width, image_height):
        proposal_transformers = proposal_transformers.view(-1, Model.NUM_CLASSES, 4)
        mean = self._transformer_normalize_mean.repeat(1, Model.NUM_CLASSES,1).cpu()
        std = self._transformer_normalize_std.repeat(1, Model.NUM_CLASSES, 1).cpu()
        
        proposal_transformers = proposal_transformers * std - mean
        proposal_bboxes = proposal_bboxes.view(-1, 1, 4).repeat(1, Model.NUM_CLASSES, 1)
        detection_bboxes = BBox.apply_transformer(proposal_bboxes.view(-1, 4), proposal_transformers.view(-1, 4))
        detection_bboxes = detection_bboxes.view(-1, Model.NUM_CLASSES, 4)
        detection_bboxes[:, :, [0,2]] = detection_bboxes[:, :, [0,2]].clamp(min = 0, max = image_width)
        detection_bboxes[:, :, [1,3]] = detection_bboxes[:,:, [1,3]].clamp(min=0, max=image_height)
        proposal_probs = F.softmax(proposal_classes, dim = 1)
        detection_bboxes = detection_bboxes.cpu()
        proposal_probs = proposal_probs.cpu()
        generated_bboxes = []
        generated_labels = []
        generated_probs = []

        for c in range(1, Model.NUM_CLASSES):
            detection_class_bboxes = detection_bboxes[:, c,:]
            proposal_class_probs = proposal_probs[:, c]

            _, sorted_indices = proposal_class_probs.sort(decending = True)
            detection_class_bboxes = detection_class_bboxes[sorted_indices]
            proposal_class_probs = proposal_class_probs[sorted_indices]
            keep_indices = NMS.suppress(detection_class_bboxes.cpu(), threshold = 0.3)
            detection_class_bboxes = detection_class_bboxes[keep_indices]
            proposal_class_probs = proposal_class_probs[keep_indices]

            generated_bboxes.append(detection_class_bboxes)
            generated_labels.append(torch.ones(len(keep_indices))* c)
            generated_probs.append(proposal_class_probs)
        generated_bboxes = torch.cat(generated_bboxes, dim = 0)
        generated_labels = torch.cat(generated_labels, dim = 0)
        generated_probs = torch.cat(generated_probs, dim = 0)
        return generated_bboxes, generated_labels, generated_probs
    
    class Detection(nn.Module):
        def __init__(self):
            super().__init__()
            self.fcs = nn.Sequential(
                nn.Linear(512 * 7 *7, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU()
            )
            self._class = nn.Linear(4096, Model.NUM_CLASSES)
            self._transformer = nn.Linear(4096, Model.NUM_CLASSES * 4)

        def forward(self, features, proposal_bboxes):
            _, _, feature_map_height, feature_map_width = features.shape
            pool = []
            for proposal_bbox in proposal_bboxes:
                start_x = max(min(round(proposal_bbox[0].item()/ 16), feature_map_width - 1),0)
                start_y = max(min(round(proposal_bbox[1].item()/ 16), feature_map_height - 1), 0)
                end_x = max(min(round(proposal_bbox[2].item()/ 16)+ 1, feature_map_width - 1),1)
                end_y = max(min(round(proposal_bbox[3].item()/ 16)+1, feature_map_height - 1), 1)
                roi_feature_map =features[..., start_y: end_y, start_x:end_x]
                pool.append(F.adaptive_max_pool2d(roi_feature_map, 7))
            pool = torch.cat(pool , dim = 0)
            pool = pool.view(pool.shape[0], -1)
            h = self.fcs(pool)
            classes = self._class(h)
            transformers = self._transformer(h)
            return classes, transformers



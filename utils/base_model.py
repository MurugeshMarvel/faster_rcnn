from typing import Type
import torchvision
from torch import nn
import sys

class BaseModel(object):
    @staticmethod
    def select_model(name):
        if name == 'vgg16':
            return Vgg16()
        elif name == 'resnet101':
            return ResNet101
        else:
            raise ValueError
    def __init__(self, pretrained: bool):
        super().__init__()
        self._pretrained = pretrained


class Vgg16():
    
    def __init__(self, pretrained= False):
        self.pretrained = pretrained
    
    def features(self):
        vgg16 = torchvision.models.vgg16(pretrained= self.pretrained)
        features = list(vgg16.features.children())[:-1]

        for parameters in [feature.parameters() for i, feature in enumerate(features) if i < 10]:
            for parameter in parameters:
                parameter.require_grad = False
        features = nn.Sequential(*features)
        return features


class ResNet101():

    def __init__(self, pretrained = False):
        self.pretrained = pretrained

    def features(self):
        resnet101 = torchvision.models.resnet101(pretrained = self.pretrained)
        features = list(resnet101.children())[:-2]
        for parameters in [feature.parameters() for i, feature in enumerate(features) if i <=6 ]:
            for parameter in parameters:
                parameter.require_grad = False
        
        features.append(nn.ConvTranspose2d(in_channels= 2048, out_channels= 512,
                                            kernel_size= 3, stride = 2, padding = 1))
        features.append(nn.ReLU())
        features = nn.Sequential(*features)
        return features
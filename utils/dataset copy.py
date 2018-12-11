import os
import random
import xml.etree.ElementTree as ET
from enum import Enum
from typing import List, Tuple
import PIL
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from torch import Tensor
from torchvision import transforms
import torch
import sys
from utils.bbox import BBox

class DataSet(Dataset):
    class Mode(Enum):
        TRAIN = 'train'
        TEST = 'test'
    
    class Annotation(object):

        class Object(object):
            def __init__(self, name, bbox):
                super().__init__()
                self.name = name
                self.bbox = bbox
            def __repr__(self):
                return 'Object[name={:s}, bbox={!s}'.format(
                    self.name, self.bbox
                )

        def __init__(self, filename:str, objects:List[object]):
            super().__init__()
            self.filename = filename
            self.objects = objects
    cat_to_lab_dict = {
        'background': 0,
        'parking_lot': 1
        }
    lab_to_cat_dict = {v: k for k,v in cat_to_lab_dict.items()}
    
    def __init__(self, data_path, mode):
        super().__init__()
        self.mode = mode
        self.image_path = os.path.join(data_path,'images')
        annotation_path = os.path.join(data_path, 'annotation')
        if self.mode == DataSet.Mode.TRAIN:
            image_file_index = os.path.join(data_path, 'train.txt')
        elif self.mode == Dataset.Mode.TEST:
            image_file_index = os.path.joint(data_path, 'test.txt')
        else:
            raise ValueError('Invalid Mode')
        with open(image_file_index, 'r') as file:
            lines = file.readlines()
            self._image_ids = [line.rstrip() for line in lines]
        self._image_id_to_annotation_dict = {}
        for img_id in self._image_ids:
            annotation_xml_file = os.path.join(annotation_path, '{}.xml'.format(img_id))
            tree = ET.ElementTree(file = annotation_xml_file)
            root = tree.getroot()
            
            self._image_id_to_annotation_dict[img_id] = DataSet.Annotation(
                filename = next(root.iterfind('filename')).text,
                objects = [DataSet.Annotation.Object(name = next(tag_object.iterfind('name')).text,
                    bbox = BBox(
                        left = float(next(tag_object.iterfind('bndbox/xmin')).text),
                        top = float(next(tag_object.iterfind('bndbox/ymin')).text),
                        right = float(next(tag_object.iterfind('bndbox/xmax')).text),
                        bottom = float(next(tag_object.iterfind('bndbox/ymax')).text))
                    )
                for tag_object in root.iterfind('object')]
            )
    def __len__(self):
        return len(self._image_id_to_annotation_dict)
    
    def __getitem__(self, index):
        img_id = self._image_ids[index]
        annotation = self._image_id_to_annotation_dict[img_id]
        bboxes = [obj.bbox.tolist() for obj in annotation.objects]
        labels = [DataSet.cat_to_lab_dict[obj.name] for obj in annotation.objects]
        bboxes = torch.tensor(bboxes, dtype = torch.float)
        labels = torch.tensor(labels, dtype = torch.long)

        image = Image.open(os.path.join(self.image_path, annotation.filename))
        if self.mode == DataSet.Mode.TRAIN and random.random() > 0.5:
            image = ImageOps.mirror(image)
            bboxes[:, [0, 2]] = image.width - bboxes[:, [2,0]]

        image, scale = DataSet.preprocess(image)
        bboxes *= scale
        return img_id, image, scale, bboxes, labels
    
    @staticmethod
    def preprocess(image):
        scale_for_shorted_edge = 600.0 / min(image.width, image.height)
        longer_edge_after_scaling = max(image.width, image.height) * scale_for_shorted_edge
        scale_for_longer_edge = (1000.0 / longer_edge_after_scaling) if longer_edge_after_scaling > 1000 else 1
        scale = scale_for_shorted_edge * scale_for_longer_edge

        transform = transforms.Compose([
            transforms.Resize((round(image.height * scale), round(image.width * scale))),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])
        image = transform(image)
        return image, scale
if __name__ == '__main__':
    def main():
        dataset = Dataset(data_path='../data', mode = Dataset.Mode.TRAIN)
        img_id, img, scale, bboxes, labels = dataset[0]
        print ('image ID', img_id)
        print ('Image Shape', img.shape)
        #print ("Image: ", img)
        print ('Scale :',scale)
        print ("bboxes Shape", bboxes.shape)
        print ("labels Shape", labels.shape)
        print (labels)
    main()
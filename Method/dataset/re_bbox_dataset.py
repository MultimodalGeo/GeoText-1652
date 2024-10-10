import json
import os
import math
import random
from random import random as rand
import torchvision.transforms as transforms
import torch

from torchvision.transforms.functional import hflip, resize

from dataset.utils import pre_caption


from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None



class re_dataset_bbox(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30, mode='train', config=None):
        self.image_res = config['image_res']

        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.mode = mode
        self.img_ids = {}

        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):
        # print('Note: This part is in the dataset building process')

        ann = self.ann[index]
        caption = pre_caption(ann['caption'], self.max_words)
        # print("Here is the caption",caption)
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        # print("Here is the original image", image)
        W, H = image.size

        # random crop
        target_bboxes = []
        sens = []
        for sen in ann["sentences"]:
            if sen is None:
                sen = 'NONE'
            else:
                sen = pre_caption(sen, self.max_words)
            sens.append(sen)
        # print("Here are the sens,",sens)
        no_bbox_value = -100
        no_bbox_tensor = [no_bbox_value, no_bbox_value, no_bbox_value, no_bbox_value]

        for box in ann["bboxes"]:
            if box is None:
                target_bboxes.append(no_bbox_tensor)
            else:

                target_bboxes.append(box)

        image = resize(image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)
        image = self.transform(image)

        target_bboxes = torch.tensor(target_bboxes, dtype=torch.float32)

        return image, caption, self.img_ids[ann['image_id']], sens, target_bboxes




class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=50):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        self.img2building = {}

        txt_id = 0
        building_id = 0
        ann_building = 0
        for img_id, ann in enumerate(self.ann):
            ann["building_id"] = ann["image_id"][:4]
            if ann_building == 0:
                ann_building = ann["building_id"]
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            self.img2building[img_id] = building_id
            if ann_building != ann["building_id"]:
                ann_building = ann["building_id"]
                building_id += 1
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index

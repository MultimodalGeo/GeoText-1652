import json
import os
import math
import random
from random import random as rand
import torchvision.transforms as transforms
import torch

from torchvision.transforms.functional import hflip, resize

from dataset.utils import pre_caption
from refTools.refer_python3 import REFER

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
                # # is_tensor = isinstance(image, torch.Tensor)
                # # if is_tensor:
                # #     to_pil = transforms.ToPILImage()
                # #     image = to_pil(image)
                # x, y, w, h = box
                # assert (x >= 0) and (y >= 0) and (x + w <= W) and (y + h <= H) and (w > 0) and (
                #         h > 0), "elem invalid"

                # x0, y0 = random.randint(0, math.floor(x)), random.randint(0, math.floor(y))
                # x1, y1 = random.randint(min(math.ceil(x + w), W), W), random.randint(min(math.ceil(y + h), H),
                #                                                                     H)  # fix bug: max -> min
                # w0, h0 = x1 - x0, y1 - y0
                # assert (x0 >= 0) and (y0 >= 0) and (x0 + w0 <= W) and (y0 + h0 <= H) and (w0 > 0) and (
                #         h0 > 0), "elem randomcrop, invalid"
                # # image = image.crop((x0, y0, x0 + w0, y0 + h0))

                # # W, H = image.size
                # # image = self.transform(image)
                # # image = test_transform(image)
                # # axis transform: for crop
                # # x = x - x0
                # # y = y - y0
                # # image = resize(image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)

                # # resize applied
                # x = self.image_res / W * x
                # w = self.image_res / W * w
                # y = self.image_res / H * y
                # h = self.image_res / H * h

                # center_x = x + 1 / 2 * w
                # center_y = y + 1 / 2 * h

                # target_bbox = [center_x / self.image_res, center_y / self.image_res,
                #                             w / self.image_res, h / self.image_res]
                target_bboxes.append(box)
        # print(target_bboxes)
        # print(caption)
        image = resize(image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)
        image = self.transform(image)
        # print(f"This is the image after transform:{image}")
        # print(image)
        target_bboxes = torch.tensor(target_bboxes, dtype=torch.float32)
        # print('Here is the target_bboxes',target_bboxes)
        # print('Here is the img_id', self.img_ids[ann['image_id']])
        return image, caption, self.img_ids[ann['image_id']], sens, target_bboxes

#注意一下， 主要矛盾产生于target_bboxes idx 和 image是深度 绑定的 而 caption 和 sens 并不是


        # else:
        #     image = self.transform(image)  # test_transform
        #     return image, caption, ann['ref_id']

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=60):
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
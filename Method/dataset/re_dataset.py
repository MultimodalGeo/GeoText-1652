import json
import os

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=50):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
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

        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = pre_caption(ann['caption'], self.max_words)

        return image, caption, self.img_ids[ann['image_id']]


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



# class re_eval_dataset(Dataset):
#     def __init__(self, ann_file, transform, image_root, max_words=30):
#         self.ann = json.load(open(ann_file, 'r'))
#         self.transform = transform
#         self.image_root = image_root
#         self.max_words = max_words

#         self.text = []
#         self.image = []
#         self.txt2img = {}
#         self.img2txt = {}
#         self.img2building = {}

#         txt_id = 0
#         for img_id, ann in enumerate(self.ann):
#             self.image.append(ann['image'])
#             self.img2txt[img_id] = []
#             for i, caption in enumerate(ann['caption']):
#                 self.text.append(pre_caption(caption, self.max_words))
#                 self.img2txt[img_id].append(txt_id)
#                 self.txt2img[txt_id] = img_id
#                 txt_id += 1

#     def __len__(self):
#         return len(self.image)

#     def __getitem__(self, index):

#         image_path = os.path.join(self.image_root, self.ann[index]['image'])
#         image = Image.open(image_path).convert('RGB')
#         image = self.transform(image)

#         return image, index
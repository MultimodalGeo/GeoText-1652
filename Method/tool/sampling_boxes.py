import sys,os
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from torchvision.ops import box_convert
import supervision as sv
import cv2
import json
import random
import numpy as np
import torch
from PIL import Image
from GroundingDINO.groundingdino.util.inference import load_image

def annotate(image_source: np.ndarray, boxes: torch.Tensor,  phrases) -> np.ndarray:
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = phrases

    box_annotator = sv.BoxAnnotator()
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    box_annotator.thinkness = 1
    box_annotator.text_scale = 0.3
    box_annotator.text_padding = 8
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame

def find_noun(sentence):
    # 分词
    words = word_tokenize(sentence)
    # 词性标注
    tagged_words = pos_tag(words)
    # 提取名词
    nouns = [word for word, pos in tagged_words if pos in ('NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS')]
    unique_nouns = list(set(nouns))
    words_to_remove = {'side', 'front', 'on', 'in', 'object', 'right', 'left', 'image', 'other', 'several'}

    # 使用列表推导式删除所有匹配的项
    unique_nouns = [word for word in unique_nouns if word not in words_to_remove]
    

    return unique_nouns

image_folder = '/storage_fast/mchu/Multi-model/geotext/GeoText'
root_path = '/storage_fast/mchu/Multi-model/geotext/GeoText/train/all_together_2_modified.json'

with open(root_path, 'r') as file:
    data = json.load(file)
    # 从data中随机选择3个字典
    random_dics = random.sample(data, 10000)

for dic in random_dics:
    img_id = dic['image_id']
    source_name = dic['image']
    source_path = os.path.join(image_folder, source_name)
    sens = dic['sentences']
    boxes = dic['bboxes']
    image_source, image = load_image(source_path)
    if None in boxes:
        continue
    t_boxes = torch.tensor(boxes)
    fine = []
    for phra in sens:
        word = find_noun(phra)
        print(word)
        fine.append(str(word))
    annotated_frame = annotate(image_source=image_source, boxes=t_boxes, phrases=fine)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB

    # 使用Image.fromarray将数组转换为图像
    image_new = Image.fromarray(annotated_frame)
    sample_folder = '/storage_fast/mchu/Multi-model/geotext/sample'
    img_id = img_id.replace('/','-')
    if img_id[-4:] != 'jpeg':
        continue
    name = img_id[:-4]+'png'
    file_path = os.path.join(sample_folder, name)
    # 使用save方法保存图像为.png文件
    image_new.save(file_path)

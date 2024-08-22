import os, sys

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

import supervision as sv

# segment anything
from segment_anything_.segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt


# diffusers
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline


from huggingface_hub import hf_hub_download


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   


ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"


groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

import re
import csv
import ast

BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0.2


def round_to_4_decimals(numbers):
    result = []
    for num in numbers:
        rounded_num = round(float(num), 4)
        result.append(rounded_num)
    return result


def summarize_image_description(description):
    # Define phrases to be removed
    phrases_to_remove = ["in the center of the image","main object","on the left side of the building","on the right side of the building"]

    # Iterate over each phrase and replace it with an empty string
    for phrase in phrases_to_remove:
        description = description.replace(phrase, "")

    # Remove multiple spaces
    description = re.sub(' +', ' ', description)

    return description.strip()  # Remove leading and trailing spaces


folder_path = "/storage_fast/mchu/Multi-model/geotext/GeoText/train"
# Open the original csv file

count = 0

for i in range(1240,1700):
    s = "%04d" % i
    path = folder_path+'/'+s+'/'+'apt.csv'
    path_ = folder_path+'/'+s+'/'+'apolla.csv'
    if os.path.exists(path)==False:
        print(s)
        continue
    else:
        print("Here is",path)
        files = []
        RECs = []
        with open(path, "r") as file:
            s = "%04d" % i
            reader = csv.reader(file)
            p = None
            q = None
            t = None
            for row in reader:
                
                # Replace all single quotes with double quotes in each row
                if row == ['image', 'prompt']:
                    continue
                else:
                    local_image_path = folder_path+"/"+s+"/"+ row[0]
                    image_source, image = load_image(local_image_path)
                    
                    new_row_0 = [cell.replace('"', "\'") for cell in row]
                    new_row = [cell.replace("['", "[\"") for cell in new_row_0]
                    new_row_1 = [cell.replace("']", '\"]') for cell in new_row]
                    new_row_2 = [cell.replace("', '", "\", \"") for cell in new_row_1]
                    new_row_3 = [cell.replace("\", '", "\", \"") for cell in new_row_2]
                    new_row_4 = [cell.replace("', \"", "\", \"") for cell in new_row_3]
                    lst = ast.literal_eval(new_row_4[1])
                    files.append(new_row_4[0])
                    # print(lst)
                    RRR = []
                    for target in lst:
                        REC={}
                        #text test for th
                        search_phrase_1 = "main object"
                        search_phrase_2 = "left"
                        search_phrase_3 = "right"

                        # Split the string into a list of sentences
                        sentences = target.split(".")
                        # print(sentences)

                        # Find the first sentence containing the search phrase
                        first_sentence = next((sentence for sentence in sentences if search_phrase_1 in sentence),None)

                        if first_sentence != None:
                            p = first_sentence
                        else:
                            first_sentence = p
                        
                        if first_sentence == None:
                            first_sentence = sentences[0]

                        second_sentence = next((sentence for sentence in sentences if search_phrase_2 in sentence),None)

                        if second_sentence != None:
                            q = second_sentence
                        else:
                            second_sentence = q
                        

                        third_sentence = next((sentence for sentence in sentences if search_phrase_3 in sentence),None)

                        if third_sentence != None:
                            t = third_sentence
                        else:
                            third_sentence = t
                            
                        if first_sentence == None or second_sentence == None or third_sentence == None:
                            continue 

                        if second_sentence == third_sentence:
                            continue

                        # print(first_sentence)

                        s1 = summarize_image_description(first_sentence)
                        s2 = summarize_image_description(second_sentence)
                        s3 = summarize_image_description(third_sentence)

                        S = [s1,s2,s3]
                        REC["object"] = S
                        B = []
                        phra = []
                        for text_prompt in S:
                            boxes, logits, phrases = predict(
                                model=groundingdino_model, 
                                image=image, 
                                caption=text_prompt, 
                                box_threshold=BOX_TRESHOLD, 
                                text_threshold=TEXT_TRESHOLD
                            )
                            # first_logit = logits[0].unsqueeze(0)
                            
                            if boxes.size() == torch.Size([0,4]):
                                continue
                                first_box = [0,0,0,0]
                                
                            else:
                                first_box = boxes[0].numpy().tolist()
                            if phrases == []:
                                continue
                                first_phrase = 'None'
                            else:
                                first_phrase = [(phrases[0])]
                            bbox = round_to_4_decimals(first_box)
                            B.append(bbox)
                            phra.append(first_phrase)
                        REC["bbox"] = B
                        REC["phrases"] = phra
                        RRR.append(REC)
                        count = count + 1
                    RECs.append(RRR)

    with open(path_, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(['image', 'REC'])
        for file, REC in zip(files, RECs):
            w.writerow([file, REC])

print('the total prase is', count)

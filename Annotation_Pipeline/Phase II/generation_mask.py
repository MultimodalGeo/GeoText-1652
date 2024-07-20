import os, sys
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
from huggingface_hub import hf_hub_download
# segment anything
from segment_anything_.segment_anything import build_sam, SamPredictor 
import numpy as np
import matplotlib.pyplot as plt

import supervision as sv
import json
import random
import numpy as np
import torch

import matplotlib.pyplot as plt
from io import BytesIO

import random as rd
import csv
import ast


def round_to_4_decimals(numbers):
    result = []
    for num in numbers:
        rounded_num = round(float(num), 4)
        result.append(rounded_num)
    return result

def compute_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x_int_left = max(x1-0.5*w1, x2-0.5*w2)
    y_int_top = max(y1-0.5*h1, y2-0.5*h2)
    x_int_right = min(x1 + 0.5*w1, x2 + 0.5*w2)
    y_int_bottom = min(y1 + 0.5*h1, y2 + 0.5*h2)

    w_int = max(0, x_int_right - x_int_left)
    h_int = max(0, y_int_bottom - y_int_top)
    area_int = w_int * h_int
    area_union = w1 * h1 + w2 * h2 - area_int

    iou = area_int / area_union if area_union > 0 else 0

    return iou


def compute_io(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x_int_left = max(x1-0.5*w1, x2-0.5*w2)
    y_int_top = max(y1-0.5*h1, y2-0.5*h2)
    x_int_right = min(x1 + 0.5*w1, x2 + 0.5*w2)
    y_int_bottom = min(y1 + 0.5*h1, y2 + 0.5*h2)

    w_int = max(0, x_int_right - x_int_left)
    h_int = max(0, y_int_bottom - y_int_top)
    area_int = w_int * h_int
    first_area = w1 * h1

    iot = area_int / first_area if first_area > 0 else 0

    return iot


def show_mask(masks, image, random_color=True):
    image_mask = []
    for i in range(len(masks)):
        mask = masks[i][0]
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        image_mask.append(mask_image)
    for i in range(len(image_mask)):
        if i == 0:
            mask_image = image_mask[0]
        else:
            mask_image = mask_image + image_mask[i]
        
       
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

# def summarize_image_description(description):
#     # Define phrases to be removed
#     phrases_to_remove = ["in the center of the image","main object","on the left side of the building","on the right side of the building"]

#     # Iterate over each phrase and replace it with an empty string
#     for phrase in phrases_to_remove:
#         description = description.replace(phrase, "")

#     # Remove multiple spaces
#     description = re.sub(' +', ' ', description)

#     return description.strip()  # Remove leading and trailing spaces

def merge_bboxes(bboxes):
    # 计算左上角和右下角坐标
    corners = [(cx - w/2, cy - h/2, cx + w/2, cy + h/2) for cx, cy, w, h in bboxes]

    # 找到左上角的最小值和右下角的最大值
    min_x = min(corners, key=lambda x: x[0])[0]
    min_y = min(corners, key=lambda x: x[1])[1]
    max_x = max(corners, key=lambda x: x[2])[2]
    max_y = max(corners, key=lambda x: x[3])[3]

    # 计算新bbox的中心坐标和宽高
    new_cx = (min_x + max_x) / 2
    new_cy = (min_y + max_y) / 2
    new_w = max_x - min_x
    new_h = max_y - min_y

    return torch.tensor([new_cx, new_cy, new_w, new_h])

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

sam_checkpoint = 'sam_vit_h_4b8939.pth'
sam = build_sam(checkpoint=sam_checkpoint)
sam.to(device='cuda')
sam_predictor = SamPredictor(sam)



groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

IOU_threhold = 0.2
IOI_threhold = 0.8
IOO_threhold = 0.9

BOX_TRESHOLD_main = 0.25
TEXT_TRESHOLD_main = 0.2

BOX_TRESHOLD_left = 0.25
TEXT_TRESHOLD_left = 0.2

BOX_TRESHOLD_right = 0.25
TEXT_TRESHOLD_right = 0.2


folder_path = "/storage_fast/mchu/Multi-model/geotext/GeoText/train"
# Open the original csv file

count_main = 0
count_left = 0
count_right = 0



for i in range(800,900):
    s = "%04d" % i
    path = folder_path+'/'+s+'/'+'apt_new.csv'
    path_ = folder_path+'/'+s+'/'+'aaf_fine_mask.csv'
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
                    print("here is the list", lst)
                    files.append(new_row_4[0])
                    # print(lst)
                    RRR = []
                    happen = 0
                    for target in lst:
                        print(target)
                        REC={}
                        REC["description"] = target
                        #text test for th
                        search_phrase_1 = "main object"
                        search_phrase_11 = "in the center of the image"
                        search_phrase_2 = "left"
                        search_phrase_3 = "right"
                    
                        # Split the string into a list of sentences
                        sentences = target.split(".")
                        # print(sentences)
                        sentences = sentences[3:]

                        # Find the first sentence containing the search phrase
                        first_sentence = next((sentence for sentence in sentences if (search_phrase_1 in sentence or search_phrase_11 in sentence)),None)

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
                            
                        if first_sentence == None or second_sentence == None or third_sentence == None or second_sentence == third_sentence:
                            REC["bbox"] = [None, None, None]
                            REC["spatial_sentences"] = [None, None, None]
                            REC['masks'] = [None, None, None]
                            RRR.append(REC)
                            continue

                        # print(first_sentence)

                        s1 = first_sentence
                        print(s1)
                        s2 = second_sentence
                        print(s2)
                        s3 = third_sentence
                        print(s3)
                        print(row[0])
                        S = [s1,s2,s3]
                        REC["spatial_sentences"] = S

                        boxes_1, logits_1, phrases_1 = predict(
                            model=groundingdino_model, 
                            image=image, 
                            caption=s1, 
                            box_threshold=BOX_TRESHOLD_main, 
                            text_threshold=TEXT_TRESHOLD_main
                        )

                        boxes_2, logits_2, phrases_2 = predict(
                            model=groundingdino_model, 
                            image=image, 
                            caption=s2, 
                            box_threshold=BOX_TRESHOLD_left, 
                            text_threshold=TEXT_TRESHOLD_left
                        )

                        boxes_3, logits_3, phrases_3 = predict(
                            model=groundingdino_model, 
                            image=image, 
                            caption=s3, 
                            box_threshold=BOX_TRESHOLD_right, 
                            text_threshold=TEXT_TRESHOLD_right
                        )

                        collect_main = []
                        main_phrases = []

                        num_box1 = len(boxes_1)
                        num_box2 = len(boxes_2)
                        num_box3 = len(boxes_3)

                        # if we hope to cut down the number of candidate
                        # if num_box1 > 3:
                        #     num_box1 = 3
                        
                        # if num_box2 > 3:
                        #     num_box2 = 3
                        
                        # if num_box3 >3:
                        #     num_box3 = 3

                        for i in range(num_box1):
                            if boxes_1[i][0]>0.3 and boxes_1[i][0]<0.7:
                                print(boxes_1[i])
                                if boxes_1[i][2]<0.95 and boxes_1[i][3]<0.95:
                                    collect_main.append(boxes_1[i])
                                    main_phrases.append(phrases_1[i])
                        

                        # print(collect_main)
                        print("Here main", collect_main)
                        if collect_main ==[]:
                            main = None
                            main_ = None
                        

                        # else:
                        #     first_logit = logits_1[0].unsqueeze(0)
                        #     main = merge_bboxes(collect_main)
                        #     if main[2] < 0.15 or main[3] < 0.15:
                        #         main = None
                        #         main_ = None
                        #     else:

                        #         print("here main",main)
                        #         main_show = main.unsqueeze(0)
                        #         main_ = main.numpy().tolist()
                        #         # main_ = round_to_4_decimals(main_)
                        #         # count_main = count_main+1
                        else:
                            first_logit = logits_1[0].unsqueeze(0)
                            OK = 1
                            num_test = 3
                            while OK:
                                main = merge_bboxes(collect_main)
                                if num_test == 0:
                                    main_ = None
                                    break
                                if main[2]<0.75 and main[3]<0.75:
                                    main_show = main.unsqueeze(0)
                                    main_ = main.numpy().tolist()
                                    OK = 0
                                else:
                                    collect_main = collect_main[:num_test]
                                    num_test = num_test-1
                            if main[2] < 0.15 or main[3] < 0.15:
                                main = None
                                main_ = None

                        collect_left = []
                        left_phrases = []
                        if main == None:
                            left_ = None
                        else:
                            for i in range(num_box2):
                                Flag = 0
                                if (boxes_2[i][0]+0.5*boxes_2[i][2])<main[0]:
                                    print(boxes_2[i])
                                    for main_box in collect_main:
                                        if compute_iou(boxes_2[i],main_box) > IOU_threhold:
                                            Flag = 1
                                            break
                                    if compute_io(boxes_2[i],main) > IOI_threhold:
                                        Flag = 1
                                    if Flag == 1:
                                        continue
                                    if boxes_2[i][2]<0.8 and boxes_2[i][3]<0.8:
                                        collect_left.append(boxes_2[i])
                                        left_phrases.append(phrases_2[i])
                            if collect_left == []:
                                left_ = None
                            else:
                                second_logit = logits_2[0].unsqueeze(0)
                                OK = 1
                                num_test = 3
                                while OK:
                                    left = merge_bboxes(collect_left)
                                    if num_test == 0:
                                        left_ = None
                                        break
                                    if left[2]<0.7 and left[3]<0.7:
                                        left_show = left.unsqueeze(0)
                                        left_ = left.numpy().tolist()
                                        OK = 0
                                        # left_ = round_to_4_decimals(left_)
                                        # count_left = count_left + 1
                                    else:
                                        collect_left = collect_left[:num_test]
                                        num_test = num_test-1
                                if left[2] < 0.15 or left[3] < 0.15:
                                    left = None
                                    left_ = None
                                    

                        collect_right = []
                        right_phrases = []
                        if main == None:
                            right_ = None
                        else:
                            for i in range(num_box3):
                                Flag = 0
                                if (boxes_3[i][0]-0.5*boxes_3[i][2])>main[0]:
                                    print(boxes_3[i])
                                    for main_box in collect_main:
                                        if compute_iou(boxes_3[i],main_box) > IOU_threhold:
                                            Flag = 1
                                            break
                                    if compute_io(boxes_3[i], main) > IOI_threhold:
                                        Flag = 1
                                    if Flag == 1:
                                        continue
                                    if boxes_3[i][2]<0.8 and boxes_3[i][3]<0.8:
                                        collect_right.append(boxes_3[i])
                                        right_phrases.append(phrases_3[i])
                            if collect_right == []:
                                right_ = None
                            else:
                                third_logit = logits_3[0].unsqueeze(0)
                                OK = 1
                                num_test = 3
                                while OK:
                                    right = merge_bboxes(collect_right)
                                    if num_test == 0:
                                        right_ = None
                                        break
                                    if right[2]<0.7 and right[3]<0.7:
                                        right_show = right.unsqueeze(0)
                                        right_ = right.numpy().tolist()
                                        OK = 0

                                    else:
                                        collect_right = collect_right[:num_test]
                                        num_test = num_test-1
                                if right[2] < 0.15 or right[3] < 0.15:
                                    right = None
                                    right_ = None
                                    

                        B = [main_,left_,right_]
                        #     # first_logit = logits[0].unsqueeze(0)
                            
                        #     if boxes.size() == torch.Size([0,4]):
                        #         continue
                        #         first_box = [0,0,0,0]
                                
                        #     else:
                        #         first_box = boxes[0].numpy().tolist()
                        #     if phrases == []:
                        #         continue
                        #         first_phrase = 'None'
                        #     else:
                        #         first_phrase = [(phrases[0])]
                        #     bbox = round_to_4_decimals(first_box)
                        #     B.append(bbox)
                        #     phra.append(first_phrase)

                        REC["bbox"] = B
                        boxes = B
                        if boxes[0] == None or (boxes[1] == None and boxes[2] == None):
                            REC['masks'] = [None, None, None]
                        elif boxes[1] == None and boxes[2] != None:
                            new_boxes = torch.tensor([boxes[0],boxes[2]])
                            sam_predictor.set_image(image_source)
                            H, W, _ = image_source.shape
                            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(new_boxes) * torch.Tensor([W, H, W, H])
                            transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to('cuda')
                            masks, _, _ = sam_predictor.predict_torch(
                                    point_coords = None,
                                    point_labels = None,
                                    boxes = transformed_boxes,
                                    multimask_output = False,
                                )
                            mask_tensor = masks.cpu().to(torch.uint8).numpy().tolist()
                            REC['masks'] = [mask_tensor[0], None, mask_tensor[1]]

                        elif boxes[2] == None and boxes[1] != None:
                            new_boxes = torch.tensor([boxes[0],boxes[1]])
                            sam_predictor.set_image(image_source)
                            H, W, _ = image_source.shape
                            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(new_boxes) * torch.Tensor([W, H, W, H])
                            transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to('cuda')
                            masks, _, _ = sam_predictor.predict_torch(
                                    point_coords = None,
                                    point_labels = None,
                                    boxes = transformed_boxes,
                                    multimask_output = False,
                                )
                            mask_tensor = masks.cpu().to(torch.uint8).numpy().tolist()
                            REC['masks'] = [mask_tensor[0], mask_tensor[1], None]

                        else:
                            boxes = torch.tensor(boxes)
                            sam_predictor.set_image(image_source)
                            H, W, _ = image_source.shape
                            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
                            transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to('cuda')
                            masks, _, _ = sam_predictor.predict_torch(
                                    point_coords = None,
                                    point_labels = None,
                                    boxes = transformed_boxes,
                                    multimask_output = False,
                                )
                            mask_tensor = masks.cpu().to(torch.uint8).numpy().tolist()
                            print('mask over!')
                            REC['masks'] = [mask_tensor[0], mask_tensor[1], mask_tensor[2]]

                        RRR.append(REC)
                        print(RRR)
                        pro = rd.random()

                        #generation and save
                        # if None not in B:
                        #     if pro<0.1 and happen==0:
                        #         print("Create Sample Image")
                        #         happen = 1
                        #         result_box = torch.cat((main_show,left_show,right_show),dim=0)
                        #         result_logit = torch.cat((first_logit,second_logit,third_logit),dim=0)

                        #         main_phrases = str(main_phrases)
                        #         left_phrases = str(left_phrases)
                        #         right_phrases = str(right_phrases)
                        #         result_pharse = [main_phrases,left_phrases,right_phrases]
                        #         annotated_frame = annotate(image_source=image_source, boxes=result_box, logits=result_logit, phrases=result_pharse)
                        #         annotated_frame = annotated_frame[...,::-1] # BGR to RGB

                        #         # 使用Image.fromarray将数组转换为图像
                        #         image_new = Image.fromarray(annotated_frame)
                        #         sample_folder = '/storage_fast/mchu/Multi-model/geotext/sample'
                        #         name = s+"-"+ row[0]
                        #         file_path = os.path.join(sample_folder, name)
                        #         # 使用save方法保存图像为.jpeg文件
                        #         image_new.save(file_path, quality=100)
                        

                    RECs.append(RRR)

    with open(path_, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(['image', 'REC'])
        for file, REC in zip(files, RECs):
            w.writerow([file, REC])

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

import csv
import os
from IPython.display import clear_output, display
from PIL import Image
from tqdm import tqdm
import json

import openai

positive = []
negative = []
neutral = []

def analyze_description(description):
    # 替换成你的 OpenAI API 密钥
    openai.api_key = ''

    # 调用 GPT-3.5 API
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=description,
      max_tokens=100
    )

    return response.choices[0].text.strip()

def check_keywords(text, positive_keywords, negative_keywords):
    for word in negative_keywords:
        if word in text:
            return "negative"
    for word in positive_keywords:
        if word in text:
            return "positive"
    return "neutral"



def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

Query = 'Please describe the [main object] in the [center of the image] with its [characteristics] and [colors] in this image. Make sure to [exclude] any [feelings], or [inferences] from the description'
Query_1 = 'Describe [spatial relationships] of [the object in the center of the image] with [other objects] in the image.[spatial relationship] includes [On the right side] of the [main object], and [On the left side] of the [main object]. Make sure to [exclude] any [feelings], or [inferences] from the description.' 

folder_path = "/storage_fast/mchu/Multi-model/try_blip2/picforvisual/shanghai" #@param {type:"string"}
max_filename_len = 128 #@param {type:"integer"}

def go_or_back(msg):
    wrong = ['img src','[image]',"I'm sorry","language model", "apologize", "?", "sorry", "spelling out", "letters", "words", "reads", "high altitude","sky","sun","<image>","https:","www."]
    spa = ["right", "left"]
    k = False
    for i in wrong:
        if i in msg:
            k = True
            break
    for i in spa:
        if i not in msg:
            k = True
            break
    return k


def sanitize_for_filename(prompt: str, max_len: int) -> str:
    name = "".join(c for c in prompt if (c.isalnum() or c in ",._-! "))
    name = name.strip()[:(max_len-4)] # extra space for extension
    return name


for i in range(0,10):
  n = i
  s = "%04d" % n
  folder_path_now = folder_path+'/'+s
  print('Here is', s)
  if os.path.exists(folder_path_now)==False:
    continue
  else:
    files = [f for f in os.listdir(folder_path_now) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')] if os.path.exists(folder_path) else []
    prompts = []
    for idx, file in enumerate(tqdm(files, desc='Generating prompts')):
        if idx > 0 and idx % 100 == 0:
            clear_output(wait=True)
        prompt = []
        i = 0
        j = 0
        while i <3:
            chat_state = CONV_VISION.copy()
            img_list = []
            chat.upload_img(os.path.join(folder_path_now, file), chat_state, img_list)
            chat.ask(Query, chat_state)
            llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=1,
                              temperature=1,
                              max_new_tokens=100,
                              max_length=2000)[0]
            chat.ask(Query_1, chat_state)
            llm_message_1 = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=1,
                              temperature=1,
                              max_new_tokens=120,
                              max_length=2000)[0]
            llm_message = llm_message +'[MOF]'+ llm_message_1
            # description = input("请输入你的描述: ")
            description_origin = llm_message
            description = 'There is a description of some buildings, please tell me if it is reasonable or unreasonable, and also tell me the reason for your judgement:' + description_origin
            # 分析描述
            analysis = analyze_description(description)
            print("GPT-3.5 分析结果:", analysis)

            # 检查关键词
            result = check_keywords(analysis, ["reasonable","Reasonable"], ["unreasonable","not reasonable"])
            if result == "positive":
                positive.append([description_origin,analysis])
                print("Thie is a reasonable description")
            elif result == "negative":
                negative.append([description_origin,analysis])
                print("This is a unreasonable description")
            else:
                neutral.append([description_origin,analysis])
                print("do not know, should negative")
            if go_or_back(llm_message) and (j<5):
                j = j+1
                continue
            else:
                if j >30:
                    llm_message = 'PROBLEM' + llm_message
                i = i +1                 
                j = 0
                prompt.append(llm_message)

        prompts.append(prompt)



        data = {'positive': positive, 'negative': negative, 'neutral': neutral}
        json_path = os.path.join(folder_path_now, 'data_positive.json')

        with open(json_path, 'w') as file:
            json.dump(data, file)


    

    if len(prompts):
        csv_path = os.path.join(folder_path_now, 'appp.csv')
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            w.writerow(['image', 'prompt'])
            for file, prompt in zip(files, prompts):
                w.writerow([file, prompt])

        print(f"\n\n\n\nGenerated {len(prompts)} prompts and saved to {csv_path}, enjoy!")
    else:
        print(f"Sorry, I couldn't find any images in {folder_path_now}")



import argparse
import os
import sys
import math

# import ruamel.yaml as yaml
from ruamel.yaml import YAML
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_re_bbox import XVLM
import logging
import gc

from models.tokenization_bert import BertTokenizer
from models.tokenization_roberta import RobertaTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
def freeze_all_except_spatial_head(model):
    for name, param in model.named_parameters():
        if name not in ['spatial_head.0.weight', 'spatial_head.0.bias', 
                        'spatial_head.1.weight', 'spatial_head.1.bias', 
                        'spatial_head.3.weight', 'spatial_head.3.bias']:
            param.requires_grad = False
        else:
            param.requires_grad = True


def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config):
    model.train()
    torch.autograd.set_detect_anomaly(True)
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_bb', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_spatial', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))    
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    for i, (image, text, idx, sens, target_bboxes) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # print('Note: This part is in the main process!')
        # print(f'here is the main {idx} image:{image}')
        # print(f'here is the main {idx} text:{text}')
        # print(f'Here is the {idx} sens:{sens}')
        # print(f'Here is the {idx} boxes:{target_bboxes}')
        sens = [list(i) for i in zip(*sens)]
        # print(f'Here is the {idx} new sens:{sens}')
        batch_length = idx.size(0)
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        caption = tokenizer(text, padding='longest', max_length=config['max_tokens'], return_tensors="pt").to(device)
        # target_bboxes 得是一个tensor ? 普通的None 没有办法存储在里面
        # text = "aerial view of university:" + text

        # 做一个 text 和 bbox的 tensor 对
        pair_text_bbox = []
        count = 0
        # flag = 0
        # print(target_bboxes)
        # print('here is a signal!')
        # print(target_bboxes[0][0])
        # print('here is another signal')
        # print(text)
        # print(sens)
        for i in range(batch_length):
            # spatial = 0
            for j in range(3):
                if target_bboxes[i][j][0] > 0:
                    target_bbox = target_bboxes[i][j]
                    # print(target_bbox)
                    sen = sens[i][j]
                    # print('Here is the sen',sen)
                    # print('Here is the target_bbox',target_bbox)
                    sen_token = tokenizer(sen, padding='longest', max_length=config['max_tokens'], return_tensors="pt").to(device)
                    count = count + 1
                    pair_text_bbox.append([i, sen_token, target_bbox])   #同序号 i 的 sen collect 在一起， 少于2两个就不管了， 让他们互相之间去比较

                

        outputs = model(image, caption.input_ids, caption.attention_mask, idx=idx, pair=pair_text_bbox)

        if len(outputs) == 2:
            loss_itc, loss_itm = outputs
            loss = loss_itc + loss_itm

        elif len(outputs) == 3:
            loss_itc, loss_itm, loss_bb = outputs
            loss = loss_itc + loss_itm + loss_bb

        elif len(outputs) == 4:
            loss_itc, loss_itm, loss_bb, loss_spatial = outputs
            loss = loss_itc + loss_itm + loss_bb + loss_spatial
        
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(f'Parameter {name} was not used during forward pass.')

        
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        scheduler.step()

        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_itc=loss_itc.item())
        metric_logger.update(loss_bb=loss_bb.item())
        if 'loss_spatial' in locals() and loss_spatial is not None:
            metric_logger.update(loss_spatial=loss_spatial.item())
        else:
            loss_spatial = None
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


# @torch.no_grad()
# def evaluation(model, data_loader, tokenizer, device, config):
#     model = model.half()
#     model = model.eval()
    
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Evaluation:'    
    
#     print('Computing features for evaluation...')
#     start_time = time.time()  

#     texts = data_loader.dataset.text   
#     num_text = len(texts)
#     text_bs = config['batch_size_test_text']  # 256
#     text_feats = []
#     text_embeds = []  
#     text_atts = []
#     for i in range(0, num_text, text_bs):
#         text = texts[i: min(num_text, i + text_bs)]
        
#         # text = "aerial view of university:" + text

#         text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
#                                return_tensors="pt").to(device)
#         text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
#         text_feat = text_output.last_hidden_state
#         text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))
#         text_embeds.append(text_embed)
#         text_feats.append(text_feat.cpu())
#         text_atts.append(text_input.attention_mask)

#     text_embeds = torch.cat(text_embeds, dim=0)
#     text_feats = torch.cat(text_feats, dim=0)
#     text_atts = torch.cat(text_atts, dim=0)

#     image_feats = []
#     image_embeds = []
#     for image, img_id in data_loader:
#         image = image.to(torch.float16)
#         image = image.to(device)

#         image_feat = model.vision_encoder(image)
#         image_embed = model.vision_proj(image_feat[:, 0, :])
#         image_embed = F.normalize(image_embed, dim=-1)

#         image_feats.append(image_feat.cpu())
#         image_embeds.append(image_embed)

#     image_feats = torch.cat(image_feats, dim=0)
#     image_embeds = torch.cat(image_embeds, dim=0)

#     print("Image embedding over ...")
    
#     sims_matrix = image_embeds @ text_embeds.t()

#     del image_embeds
#     del text_embeds

#     gc.collect()
#     torch.cuda.empty_cache()
#     print(f"here is the image length {len(data_loader.dataset.image)}")
#     print(f"here is the text length {len(texts)}")

#     # score_matrix_i2t = torch.full((len(data_loader.dataset.image), len(texts)), -100.0, dtype=torch.float16).to(device)
#     score_matrix_i2t = torch.full((len(texts), len(data_loader.dataset.image)), -100.0, dtype=torch.float16).to(device)

#     num_tasks = utils.get_world_size()
#     rank = utils.get_rank()
#     step = sims_matrix.size(0) // num_tasks + 1
#     start = rank * step
#     end = min(sims_matrix.size(0), start + step)

#     for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
#         topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
#         a = start + i
#         # print('Here is the a', a)
#         # print(image_feats)
#         # print(image_feats[start+i].repeat(256,1,1))

#         encoder_output = image_feats[start + i].repeat(config['k_test'], 1, 1).to(device)
#         encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
#         output = model.text_encoder(encoder_embeds=text_feats[topk_idx].to(device),
#                                     attention_mask=text_atts[topk_idx].to(device),
#                                     encoder_hidden_states=encoder_output,
#                                     encoder_attention_mask=encoder_att,
#                                     return_dict=True,
#                                     mode='fusion'
#                                     )
#         score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
#         # print("score fp16",score)
#         # fp32_tensor = score.to(torch.float32)

#         # # 找到最大绝对值，用于计算缩放因子
#         # max_val = torch.max(torch.abs(fp32_tensor))

#         # # 计算缩放因子，确保缩放后的值不超过127
#         # scale_factor = 127 / max_val

#         # # 缩放张量
#         # scaled_tensor = torch.round(fp32_tensor * scale_factor)

#         # # 类型转换到Int8
#         # score = scaled_tensor.type(torch.int8)
#         # print("score int8 shape,", score.shape)
#         # print("score int8", score)
#         # print("score_matrix_i2t,", score_matrix_i2t.shape)
#         # print('topk_idx,', topk_idx)
        

#         score_matrix_i2t[start + i, topk_idx] = score

#     #compute the other one 
#     if args.distributed:
#         dist.barrier()   
#         torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 

#     score_matrix_i2t_c = score_matrix_i2t.cpu().numpy()
#     del score_matrix_i2t
#     torch.cuda.empty_cache()
#     print("i2t score over ...")

#     sims_matrix = sims_matrix.t()
#     score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0, dtype=torch.float16).to(device)
    
#     step = sims_matrix.size(0)//num_tasks + 1
#     start = rank*step
#     end = min(sims_matrix.size(0), start + step)

#     for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
#         topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
#         encoder_output = image_feats[topk_idx].to(device)
#         encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
#         output = model.text_encoder(encoder_embeds=text_feats[start + i].repeat(config['k_test'], 1, 1),
#                                     attention_mask=text_atts[start + i].repeat(config['k_test'], 1),
#                                     encoder_hidden_states=encoder_output,
#                                     encoder_attention_mask=encoder_att,
#                                     return_dict=True,
#                                     mode='fusion'
#                                     )
#         score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
#         score = score.to(torch.float16)
#         score_matrix_t2i[start + i, topk_idx] = score

#     if args.distributed:
#         dist.barrier()   

#         torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        
        
#     total_time = time.time() - start_time
#     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#     print('Evaluation time {}'.format(total_time_str)) 

#     return score_matrix_i2t_c, score_matrix_t2i.cpu().numpy()

@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    model = model.half()
    model = model.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = config['batch_size_test_text']  # 256
    text_feats = []
    text_embeds = []  
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        
        # text = "aerial view of university:" + text

        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))
        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    image_feats = []
    image_embeds = []
    for image, img_id in data_loader:
        image = image.to(torch.float16)
        image = image.to(device)
        image_feat = model.vision_encoder(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)

        image_feats.append(image_feat)
        image_embeds.append(image_embed)

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    print("Image embedding over ...")
    
    sims_matrix = image_embeds @ text_embeds.t()

    del image_embeds
    del text_embeds

    gc.collect()




    score_matrix_i2t = torch.full((len(data_loader.dataset.image), len(texts)), -100.0, dtype=torch.float16).to(device)

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_feats[start + i].repeat(config['k_test'], 1, 1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds=text_feats[topk_idx],
                                    attention_mask=text_atts[topk_idx],
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    mode='fusion'
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[start + i, topk_idx] = score
    #compute the other one 
    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 

    score_matrix_i2t_c = score_matrix_i2t.cpu().numpy()
    del score_matrix_i2t
    torch.cuda.empty_cache()
    print("i2t score over ...")





    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0, dtype=torch.float16).to(device)
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds=text_feats[start + i].repeat(config['k_test'], 1, 1),
                                    attention_mask=text_atts[start + i].repeat(config['k_test'], 1),
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    mode='fusion'
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score

    if args.distributed:
        dist.barrier()   

        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return score_matrix_i2t_c, score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt,img2building):

    # print(scores_i2t.shape)
    # print(scores_i2t)
    # print(scores_t2i.shape)
    # print(scores_t2i)
    # print(txt2img)
    # print(img2txt)
    # print(img2building)
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        for i in range(len(inds)):
            inds[i] = img2building[txt2img[inds[i]]]
        target = np.where(inds == img2building[index])[0]
        ranks[index] = target[0]

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        for i in range(len(inds)):
            inds[i] = img2building[inds[i]]
        building = img2building[txt2img[index]]            
        ranks[index] = np.where(inds == building)[0][0]
    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean}
    return eval_result

def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    world_size = utils.get_world_size()

    if args.bs > 0:
        config['batch_size_train'] = args.bs // world_size

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating model", flush=True)
    model = XVLM(config=config)
    model.load_pretrained(args.checkpoint, config, is_eval=args.evaluate)
    if args.evaluate:
        model = model.half()   #每次evaluate 要开
    model = model.to(device)
    # if not args.evaluate:
    #     freeze_all_except_spatial_head(model)

    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu]
        ,find_unused_parameters=True
        )    #change something
        model_without_ddp = model.module

    if config['use_roberta']:
        tokenizer = RobertaTokenizer.from_pretrained(config['text_encoder'])
    else:
        tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])

    print("Creating retrieval dataset", flush=True)
    train_dataset, test_dataset = create_dataset('re_bbox', config, args.evaluate)

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)

    if args.evaluate:
        print("Start evaluating", flush=True)
        test_loader = create_loader([test_dataset], [None],
                                    batch_size=[config['batch_size_test']],
                                    num_workers=[0],
                                    is_trains=[False],
                                    collate_fns=[None])[0]

        # score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, tokenizer, device, config)
        score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, tokenizer, device, config)

        if utils.is_main_process():
            # val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)
            # print(val_result)
            test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt, test_loader.dataset.img2building)
            print(test_result)

        dist.barrier()

    else:
        print("Start training", flush=True)

        train_dataset_size = len(train_dataset)

        if utils.is_main_process():
            print(f"### data {train_dataset_size}, batch size, {config['batch_size_train']} x {world_size}")

        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [ None]
        else:
            samplers = [None, None]

        train_loader, test_loader = create_loader([train_dataset, test_dataset], samplers,
                                                              batch_size=[config['batch_size_train']] + [
                                                                  config['batch_size_test']],
                                                              num_workers=[0, 0],
                                                              is_trains=[True, False],
                                                              collate_fns=[None, None])

        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size/(config['batch_size_train']*world_size))
        lr_scheduler = create_scheduler(arg_sche, optimizer)

        max_epoch = config['schedular']['epochs']
        best = 0
        best_epoch = 0

        for epoch in range(0, max_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            print('here is the trainloader!')
            print(train_loader)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, device, lr_scheduler, config)

            # score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, tokenizer, device, config)
            # score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, tokenizer, device, config)

            if utils.is_main_process():
                # val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt, val_loader.dataset.img2building)
                # print(val_result)
                # test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt, test_loader.dataset.img2building)
                # print(test_result)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            #  **{f'val_{k}': v for k, v in val_result.items()},
                            #  **{f'test_{k}': v for k, v in test_result.items()},
                             'epoch': epoch}

                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # if test_result['r_mean'] > best:
                #     save_obj = {
                #         'model': model_without_ddp.state_dict(),
                #         # 'optimizer': optimizer.state_dict(),
                #         # 'lr_scheduler': lr_scheduler.state_dict(),
                #         'config': config,
                #         # 'epoch': epoch,
                #     }
                #     torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                #     best = test_result['r_mean']
                #     best_epoch = epoch

                if epoch <= config['schedular']['epochs'] - 1:
                    # if (epoch+1)%2 ==0:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint_{epoch}.pth'))

            dist.barrier()
            torch.cuda.empty_cache()

        if utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write("best epoch: %d" % best_epoch)

            os.system(f"cat {args.output_dir}/log.txt")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)  # this script works for both mscoco and flickr30k
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')

    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus")
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()

    yaml = YAML(typ='rt')  # Create a YAML object with round-trip type
    with open(args.config, 'r') as file:
        config = yaml.load(file)


    # config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    

    main(args, config)
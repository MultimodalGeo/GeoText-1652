import itertools
import torch
from models import XVLMBase, load_pretrained

class BBoxCollector:
    # ... [与之前的BBoxCollector定义相同] ...
    # 记得在calculate_loss()返回spatial_loss时，你可能需要进行相应的调整以确保其正确返回。
    def __init__(self):
        self.collect_bbox = []
        self.current_num = None

    def update_bbox(self, bbox_info):
        new_num = bbox_info['num']

        # 情况1
        if not self.collect_bbox:
            self.collect_bbox.append(bbox_info)
            self.current_num = new_num
            return

        # 情况2
        if len(self.collect_bbox) == 1:
            if new_num == self.current_num:
                self.collect_bbox.append(bbox_info)
                return
            else:
                # 排出旧的bbox
                self.collect_bbox = [bbox_info]
                self.current_num = new_num
                return

        if len(self.collect_bbox) == 2:
            if new_num == self.current_num:
                self.calculate_loss(self.collect_bbox)
                self.collect_bbox = []  # 清空
                self.collect_bbox.append(bbox_info)
            else:
                self.calculate_loss(self.collect_bbox)
                self.collect_bbox = []  # 清空
                self.collect_bbox.append(bbox_info)
                self.current_num = new_num

    def calculate_loss(self, bboxes):
        permutations = list(itertools.permutations(bboxes, 2))
        for pair in permutations:
            target_bbox_A = pair[0]['bbox']
            target_bbox_B = pair[1]['bbox']
            
            sen_token_A = pair[0]['text_token']
            sen_embeds_A = pair[0]['text_embeds']
            sen_token_B = pair[1]['text_token']
            sen_embeds_B = pair[1]['text_embeds']
            
            feature_map = pair[0]['image_feature_map']  # 仅使用第一个bbox的feature map，您可能需要进行相应的调整
            
            target_ids = compute_rela(target_bbox_A, target_bbox_B)
            spatial_loss = self.get_spatial_relation_loss(sen_token_A, sen_embeds_A, sen_token_B, sen_embeds_B, target_bbox_A, target_bbox_B, feature_map, target_ids)
            print("Calculated spatial loss:", spatial_loss)

def compute_rela(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    len_x = x1 - x2
    a_len_x = abs(len_x)

    len_y = y1 - y2
    a_len_y = abs(len_y)

    if a_len_x < 0.5 * w1:
        if len_y > 0:
            return torch.tensor([0, -1])
        if len_y < 0:
            return torch.tensor([0, 1])
    
    else:
        if len_x > 0:
            if a_len_y < 0.5 * h1:
                return torch.tensor([-1, 0])
            else:
                if len_y > 0:
                    return torch.tensor([-1, -1])
                else:
                    return torch.tensor([-1, 1])
        
        if len_x < 0:
            if a_len_y < 0.5 * h1:
                return  torch.tensor([1, 0])
            else:
                if len_y > 0:
                    return torch.tensor([1, -1])
                else:
                    return torch.tensor([1, 1])



class XVLM(XVLMBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=False, use_bbox_loss=True)

        self.num_attention_heads = self.text_encoder.config.num_attention_heads
        self.init_params = []

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def forward(self, image, text_ids, text_atts, idx=None, pair=None):
        # print("Note: This part is in the model process!")
        # print(f"Here is the model {idx} image: {image}")
        # print(f'Here is the model {idx} text_ids:{text_ids}')
        # print(f'Here is the model {idx} text_atts:{text_atts}')
        # print(f'Here is the model {idx} pair:{pair}')
        image_embeds, image_atts = self.get_vision_embeds(image)
        # print('Here is the image_embeding size')
        # print(image_embeds.size(0))
        text_embeds = self.get_text_embeds(text_ids, text_atts)
        # output_coord & target_bbox: 64, 4
        image_feat, text_feat = self.get_features(image_embeds, text_embeds)

        loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx=idx)
        loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat, idx=idx)

        # print(f'loss_itc is {loss_itc}, loss_itm is {loss_itm}')
        n = len(pair)
        # print(f"the length of the pair is:{n}")
        if n == 0:
            # loss_bb = -100
            return loss_itc, loss_itm
        else:
            total_spatial_loss = 0.0  # 用于累积空间关系损失
            loss_count = 0  # 用于记录计算出的空间关系损失的数量
            for i in range(n):
                loss_bb = 0
                repeat_image = 0
                num = pair[i][0]
                new = image[num].unsqueeze(0)
                image_embeds, _ = self.get_vision_embeds(new)

                vis = self.vision_encoder.forward(new,feature_map=12)
                feature_map = vis.permute(0,3,1,2)
                
                sen_token = pair[i][1]
                sen_embeds = self.get_text_embeds(sen_token.input_ids, sen_token.attention_mask)
                print(f'Here is the number{pair[i][0]}')
                # print(sen_embeds.size(0))
                # print(image_embeds.size(0))
                output_coord = self.predict_bbox(image_embeds, sen_embeds, sen_token.attention_mask)
                # print('let us see the pair')
                # print(pair[i][2])
                # print(pair[i][2].size())
                # print(output_coord)
                # print(output_coord.size())
                loss_bbox, loss_giou = self.get_bbox_loss(output_coord, pair[i][2].unsqueeze(0))
                loss_bb += (loss_bbox + loss_giou)
                # Update the BBoxCollector with the current bbox information from the pair
                bbox_info = {
                    'text_token': pair[i][1],  # assuming this is sen_embeds
                    'text_embeds': pair[i][1],  # assuming this is sen_embeds
                    'bbox': pair[i][2],  # bbox
                    'image_feature_map': feature_map,
                    'num': pair[i][0]  # num
                }
                spatial_loss = self.bbox_collector.update_bbox(bbox_info)

                if spatial_loss is not None:  # 如果计算出了空间关系损失
                    total_spatial_loss += spatial_loss
                    loss_count += 1

            if loss_count > 0:
                average_spatial_loss = total_spatial_loss / loss_count
            else:
                average_spatial_loss = 0.0
                # print(f"loss_bb is {loss_bb}")
            loss_bb = loss_bb/n
            # print(f"loss_bb is {loss_bb}")
            return loss_itc, loss_itm, loss_bb
        

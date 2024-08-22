import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.re_dataset import re_train_dataset, re_eval_dataset
from dataset.pretrain_dataset import ImageTextJsonDataset, RegionTextJsonDataset
from dataset.nlvr_dataset import nlvr_dataset
from dataset.vqa_dataset import vqa_dataset
from dataset.grounding_dataset import grounding_dataset, grounding_dataset_bbox
from dataset.coco_karpathy_dataset import coco_karpathy_train, coco_karpathy_train_scst, coco_karpathy_caption_eval
from dataset.re_bbox_dataset import re_dataset_bbox

from dataset.randaugment import RandomAugment
#transform 去除了对于图像的水平旋转和镜像

'''
RandomResizedCrop:此操作首先随机调整图像的大小和纵横比,然后对其进行裁剪。结果图像的大小会是config['image_res']。scale=(0.2, 1.0)表示图像的原始面积与裁剪面积之间的比例范围,这里是0.2到1.0之间。interpolation=Image.BICUBIC表示在缩放图像时使用双三次插值。

RandomHorizontalFlip:以50%的概率对图像进行水平翻转。

RandomAugment:这是一种数据增强方法,但并不是torchvision库中的标准方法,可能是某个特定库或自定义实现的。从其参数可以看出,它将在给定的数据增强方法列表augs中随机选择2种方法,并应用最多7次的随机增强。所列出的增强方法包括:

Identity:保持不变
AutoContrast:自动调整图像的对比度
Equalize:使图像的直方图均衡化
Brightness:调整图像的亮度
Sharpness:调整图像的锐度
ShearX、ShearY:对图像进行X轴或Y轴的扭曲
TranslateX、TranslateY:对图像在X轴或Y轴上进行随机平移
Rotate 随机旋转图像
ToTensor 将PIL Image或numpy ndarray的图像数据转换为torch.Tensor 并且调整其范围从[0, 255]到[0.0, 1.0]。

normalize 对图像数据进行归一化。具体的均值和标准差应该在normalize变量中定义 但您没有提供这部分代码。归一化有助于神经网络的训练稳定和加速。
'''
def create_dataset(dataset, config, evaluate=False):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    pretrain_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.2, 1.0),
                                     interpolation=Image.BICUBIC),
        # transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness']),
        transforms.ToTensor(),
        normalize,
    ])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0),
                                     interpolation=Image.BICUBIC),
        # transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness']),
        transforms.ToTensor(),
        normalize,
    ])

    train_transform_wohflip = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0),
                                     interpolation=Image.BICUBIC),
        # transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',]),
        transforms.ToTensor(),
        normalize,
    ])

    box_transform = transforms.Compose([
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness']),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset == 'pretrain':
        general_dataset = ImageTextJsonDataset(config, config['train_file'], rank=int(os.environ.get('RANK') or 0),
                                               world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True,
                                               transform=pretrain_transform)

        region_dataset = RegionTextJsonDataset(config, config['train_file_regions'], rank=int(os.environ.get('RANK') or 0),
                                                world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True,
                                                transform=pretrain_transform, box_transform=box_transform)

        return general_dataset, region_dataset

    elif dataset == 're':
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        if evaluate:
            return None, test_dataset

        train_dataset = re_train_dataset(config['train_file'], train_transform, config['image_root'])
        
        # val_dataset = re_eval_dataset(config['val_file'], test_transform, config['image_root'])
        return train_dataset, test_dataset
    #修改

    elif dataset == 're_bbox':
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        if evaluate:
            return None, test_dataset

        train_transform = transforms.Compose([
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness']),
            transforms.ToTensor(),
            normalize,
        ])
        
        train_dataset = re_dataset_bbox(config['train_file'], train_transform, config['image_root'], mode='train', config=config)
        return train_dataset, test_dataset

    elif dataset == 'vqa':
        vqa_test_dataset = vqa_dataset(config['test_file'], test_transform, config['vqa_root'], config['vg_root'],
                                       split='test', answer_list=config['answer_list'],
                                       text_encoder=config['text_encoder'], use_roberta=config['use_roberta'])
        if evaluate:
            return None, vqa_test_dataset

        train_dataset = vqa_dataset(config['train_file'], train_transform_wohflip, config['vqa_root'], config['vg_root'],
                                    split='train', text_encoder=config['text_encoder'], use_roberta=config['use_roberta'])
        return train_dataset, vqa_test_dataset

    elif dataset == 'nlvr_pretrain':
        general_dataset = ImageTextJsonDataset(config, config['train_file'], rank=int(os.environ.get('RANK') or 0),
                                               world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True,
                                               transform=pretrain_transform)

        return general_dataset

    elif dataset == 'nlvr':
        test_dataset = nlvr_dataset(config['test_file'], test_transform, config['image_root'])
        if evaluate:
            return None, None, test_dataset

        train_dataset = nlvr_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = nlvr_dataset(config['val_file'], test_transform, config['image_root'])
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'grounding':
        test_dataset = grounding_dataset(config['test_file'], test_transform, config['image_root'], mode='test')
        if evaluate:
            return None, test_dataset

        train_transform = transforms.Compose([
            transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness']),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = grounding_dataset(config['train_file'], train_transform, config['image_root'], mode='train')
        return train_dataset, test_dataset

    elif dataset == 'grounding_bbox_pretrain':
        region_dataset = RegionTextJsonDataset(config, config['train_file_regions'], rank=int(os.environ.get('RANK') or 0),
                                                world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True,
                                                transform=pretrain_transform, box_transform=box_transform)

        return region_dataset

    elif dataset == 'grounding_bbox':
        test_dataset = grounding_dataset_bbox(config['test_file'], test_transform, config['image_root'], mode='test', config=config)
        if evaluate:
            return None, test_dataset

        train_transform = transforms.Compose([
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness']),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = grounding_dataset_bbox(config['train_file'], train_transform, config['image_root'], mode='train', config=config)
        return train_dataset, test_dataset

    elif dataset == 'captioning_pretrain':
        general_dataset = ImageTextJsonDataset(config, config['train_file'], rank=int(os.environ.get('RANK') or 0),
                                               world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True,
                                               transform=pretrain_transform, add_eos=True)
        return general_dataset

    elif dataset == 'caption_coco':
        train_dataset = coco_karpathy_train(train_transform, config['image_root'], config['train_file'], prompt=config['prompt'], max_words=config['max_tokens'])
        val_dataset = coco_karpathy_caption_eval(test_transform, config['image_root'], config['val_file'], 'val')
        test_dataset = coco_karpathy_caption_eval(test_transform, config['image_root'], config['test_file'], 'test')

        return train_dataset, val_dataset, test_dataset

    elif dataset == 'caption_coco_scst':
        train_dataset = coco_karpathy_train_scst(train_transform, config['image_root'], config['train_file'],
                                            prompt=config['prompt'], max_words=config['max_tokens'])
        val_dataset = coco_karpathy_caption_eval(test_transform, config['image_root'], config['val_file'], 'val')
        test_dataset = coco_karpathy_caption_eval(test_transform, config['image_root'], config['test_file'], 'test')

        return train_dataset, val_dataset, test_dataset

    else:
        raise NotImplementedError(f"dataset == {dataset}")


def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list, dim=0), question_list, answer_list, torch.Tensor(weight_list), n


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)

    if len(loaders) <= 1:
        print(f"### be careful: func create_loader returns a list length of {len(loaders)}")

    return loaders

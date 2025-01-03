import argparse
import itertools
import json
import os
import random
import subprocess
import time
from functools import partial
from typing import Optional
import torch
from internvl.model import load_seg_model_and_tokenizer, load_mag_model_and_tokenizer
from PIL import Image
from textvqa_eval import TextVQAAccuracyEvaluator
from tqdm import tqdm
from typing import Dict, Literal, Optional
import cv2
from internvl.model.segment_anything import ResizeLongestSide
from internvl.train.constants import (
    IMG_END_TOKEN,
    SEG_TOKEN,
)
import ast
import torch.nn.functional as F
from copy import deepcopy
from internvl.patch import get_mask_from_data
from internvl.train.dataset import (
    build_transform,
    dynamic_preprocess,
    preprocess_internvl2_5_seg,
)
from internvl.patch.pad_data_collator import concat_seg_data_collator
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from internvl.utils.solver import eval_seg


ds_collections = {
    'color_block': {
        # 'test': '/cpfs01/user/caixinyu/chenxiangnan/dsw/InternVL_SEG/internvl_chat/color_block_content_qa.jsonl',
        'test': '/cpfs01/user/caixinyu/markdownGenerate/test_bbox/color_block_content_qa.jsonl',
        # 'test': '/cpfs01/user/caixinyu/chenxiangnan/dsw/InternVL_SEG/internvl_chat/shell/data/color_test.jsonl',
    },
    'vqav2_testdev': {
        'train': 'data/vqav2/vqav2_train.jsonl',
        'test': 'data/vqav2/vqav2_testdev.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'okvqa_val': {
        'train': 'data/okvqa/okvqa_train.jsonl',
        'test': 'data/okvqa/okvqa_val.jsonl',
        'question': 'data/okvqa/OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'data/okvqa/mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'textvqa_val': {
        'train': 'data/textvqa/textvqa_train.jsonl',
        'test': 'data/textvqa/textvqa_val.jsonl',
        'question': 'data/textvqa/textvqa_val_questions.json',
        'annotation': 'data/textvqa/textvqa_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'textvqa_val_ocr': {
        'train': 'data/textvqa/textvqa_train.jsonl',
        'test': 'data/textvqa/textvqa_val_llava.jsonl',
        'question': 'data/textvqa/textvqa_val_questions.json',
        'annotation': 'data/textvqa/textvqa_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_val': {
        'train': 'data/vizwiz/vizwiz_train.jsonl',
        'test': 'data/vizwiz/vizwiz_val.jsonl',
        'question': 'data/vizwiz/vizwiz_val_questions.json',
        'annotation': 'data/vizwiz/vizwiz_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_test': {
        'train': 'data/vizwiz/vizwiz_train.jsonl',
        'test': 'data/vizwiz/vizwiz_test.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'docvqa_val': {
        'train': 'data/docvqa/train.jsonl',
        'test': 'data/docvqa/val.jsonl',
        'annotation': 'data/docvqa/val/val_v1.0.json',
        'metric': 'anls',
        'max_new_tokens': 100,
    },
    'docvqa_test': {
        'train': 'data/docvqa/train.jsonl',
        'test': 'data/docvqa/test.jsonl',
        'metric': None,
        'max_new_tokens': 100,
    },
    'chartqa_test_human': {
        'train': 'data/chartqa/train_human.jsonl',
        'test': 'data/chartqa/test_human.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'chartqa_test_augmented': {
        'train': 'data/chartqa/train_augmented.jsonl',
        'test': 'data/chartqa/test_augmented.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'gqa_testdev': {
        'train': 'data/gqa/train.jsonl',
        'test': 'data/gqa/test_balanced.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'gqa_testdev_llava': {
        'train': 'data/gqa/train.jsonl',
        'test': 'data/gqa/llava_gqa_testdev_balanced_qwen_format.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'ocrvqa_val': {
        'train': 'data/ocrvqa/ocrvqa_train.jsonl',
        'test': 'data/ocrvqa/ocrvqa_val.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ocrvqa_test': {
        'train': 'data/ocrvqa/ocrvqa_train.jsonl',
        'test': 'data/ocrvqa/ocrvqa_test.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ai2diagram_test': {
        'train': 'data/ai2diagram/train.jsonl',
        'test': 'data/ai2diagram/test_vlmevalkit.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'infographicsvqa_val': {
        'train': 'data/infographicsvqa/train.jsonl',
        'test': 'data/infographicsvqa/val.jsonl',
        'annotation': 'data/infographicsvqa/infographicsVQA_val_v1.0_withQT.json',
        'metric': 'anls',
        'max_new_tokens': 100,
    },
    'infographicsvqa_test': {
        'train': 'data/infographicsvqa/train.jsonl',
        'test': 'data/infographicsvqa/test.jsonl',
        'annotation': 'data/infographicsvqa/infographicsVQA_test_v1.0.json',
        'metric': None,
        'max_new_tokens': 100,
    }
}



class ValDataset(torch.utils.data.Dataset):

    def __init__(self, test, image_size=448, dynamic_image_size=False,
                 use_thumbnail=False, max_num=12, train=None, sam_size=1024, tokenizer = None, num_image_token = None):
        self.test = open(test).readlines()[-200:]
        # self.test = open(test).readlines()
        self.image_size = image_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=image_size)
        self.sam_size = sam_size
        self.transform_sam = ResizeLongestSide(sam_size) #Image size of segmentation model
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.ignore_label = 255
        self.template_name = 'internvl2_5_seg'
        self.tokenizer = tokenizer
        self.num_image_token = num_image_token

    def __len__(self):
        return len(self.test)
    def get_preprocess_function(self):
        return preprocess_internvl2_5_seg
        
    def preprocess_seg_answer(self, data_item):
        #--后续还是在数据构建的时候，就构建seg的对话形式比较好
        # bbox 列表
        bbox = data_item['bbox']  
        bbox = ast.literal_eval(bbox)
        data_item['bbox'] = bbox
        # 根据列表长度生成 [SEG] 内容
        seg_list = [SEG_TOKEN] * len(bbox)
        # 格式化输出
        if len(seg_list) == 1:
            result = f"Based on the question, the queried region is {seg_list[0]}."
        elif len(seg_list) == 2:
            result = f"Based on the question, the queried regions are {seg_list[0]} and {seg_list[1]}."
        else:
            result = f"Based on the question, the queried regions are {', '.join(seg_list[:-1])}, and {seg_list[-1]}."
        data_item['conversations'][1]['value'] = result
        return data_item
    def sam_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.sam_size - h
        padw = self.sam_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    def load_image(self, image_path):
        return Image.open(image_path).convert('RGB')

    def __getitem__(self, idx):
        data_item = json.loads(self.test[idx].strip())
        # bbox = data_item['bbox']  
        # bbox = ast.literal_eval(bbox)
        # data_item['bbox'] = bbox

        data_item = self.preprocess_seg_answer(data_item)
        #----
        # Ensure the first conversation contains an image placeholder
        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

        # Merge the image path
        image_path = data_item['image']
        # Load the image using tcs_loader if available, otherwise use PIL
        image = self.load_image(image_path)
        orig_size = image.size

        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(image, min_num=1, max_num=12,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:  # Otherwise, use the original image as a single patch
            images = [image]

        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()
        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches],
                                  group_by_length=True)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == 1, f'image tokens are truncated, this dataset is {self.ds_name}'
        #---- 给SAM 制造输入----------------------------
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_sam = self.transform_sam.apply_image(image)
        resize_sam = image_sam.shape[:2]
        image_sam = self.sam_preprocess(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())
        conversations = [data_item['conversations']]
        masks = get_mask_from_data(data_item, image)
        masks = torch.from_numpy(masks).unsqueeze(0)
        labels_sam = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        # Create the final return dictionary
        ret = dict(
            #--VLM 训练使用
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            #--SAM 分割使用
            images = image_sam, 
            conversations = conversations,
            resize_list = resize_sam,
            do_segs = True,
            masks_list = masks,
            label_list = labels_sam,
            inference = True,
        )
        return ret

       
def _prepare_input(data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """ 
        if isinstance(data, Mapping):
            return type(data)({k: _prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(_prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": 0}
            return data.to(**kwargs)
        return data

class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)




def evaluate_chat_model():
    random.seed(args.seed)
    for ds_name in args.datasets:
        dataset = ValDataset(
            test=ds_collections[ds_name]['test'],
            image_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num,
            tokenizer=tokenizer,
            num_image_token = model.num_image_token,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(concat_seg_data_collator, tokenizer=tokenizer),
        )
        # giou, ciou = eval_seg(dataloader, model, None, None)
        # print(f"evaluate in {ds_name}. giou: {giou:.4f}, ciou: {ciou:.4f}")
        giou, ciou = eval_seg(dataloader, mag_model, None, None)
        print(f"evaluate in {ds_name}. giou: {giou:.4f}, ciou: {ciou:.4f}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/cpfs01/user/caixinyu/chenxiangnan/dsw/InternVL_SEG/internvl_chat/work_dirs/InternVL2_5-1B_seg_2w_color')
    parser.add_argument('--datasets', type=str,
                        default='color_block')
    parser.add_argument('--base_path', type=str,
                        default='/cpfs01/user/caixinyu/lisiqi/InternVL/internvl_chat/pretrained/InternVL2_5-1B')
    parser.add_argument('--segmentation_model_path', type=str,
                        default='/cpfs01/shared/ADLab/hug_ckpts/chenxiangnan/sam_vit_h_4b8939.pth')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--few-shot', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true', default=True)
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--seg_model', type=str, default='work_dirs/InternVL2_5-1B_seg_8w_color')
    parser.add_argument('--mixed_strategy', type=str, default='only_seg')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=3328)
    parser.add_argument('--embedding_dim', type=int, default=4096)
    parser.add_argument('--seg_tokens', type=int, default=64)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    model, tokenizer = load_seg_model_and_tokenizer(args)
    mag_model, mag_tokenizer = load_mag_model_and_tokenizer(args)
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')

    evaluate_chat_model()

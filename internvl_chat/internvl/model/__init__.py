# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import math

import torch
from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel
from transformers import AutoTokenizer
from internvl.model.internvl_seg import (
    SegInternVLForCausalLM,
    init_vision_seg_for_model,
    InternVLMagnifier,
)
from internvl.train.constants import SEG_TOKEN, IMG_CONTEXT_TOKEN
from safetensors.torch import load_file
import os


def split_model(num_layers, vit_alpha=0.5):
    device_map = {}
    world_size = torch.cuda.device_count()
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - vit_alpha))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * (1 - vit_alpha))
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


def load_model_and_tokenizer(args):
    if args.auto:
        config = InternVLChatConfig.from_pretrained(args.checkpoint)
        num_hidden_layers = config.llm_config.num_hidden_layers
        device_map = split_model(num_hidden_layers)
    kwargs = {'device_map': device_map} if args.auto else {}
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        args.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit, **kwargs).eval()
    if not args.load_in_8bit and not args.load_in_4bit and not args.auto:
        model = model.cuda()
    return model, tokenizer

def load_seg_model_and_tokenizer(args):
    checkpoint = args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, use_fast=False)
    config = InternVLChatConfig.from_pretrained(checkpoint)
    seg_model_args = {
        "seg_token_idx": tokenizer(SEG_TOKEN, add_special_tokens=False).input_ids[0],
        "segmentation_model_path": args.segmentation_model_path,
        "tokenizer": tokenizer,
    }
    #--from_pretrained 已经正确加载模型权重了。保险起见再用safetensor加载一遍
    model = SegInternVLForCausalLM.from_pretrained(
        checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,config=config, **seg_model_args)
    # 加载权重到模型
    file_path = os.path.join(checkpoint, 'model.safetensors')
    state_dict = load_file(file_path)
    model.load_state_dict(state_dict, strict=False)
    if not args.load_in_8bit and not args.load_in_4bit and not args.auto:
        model = model.cuda()
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    return model, tokenizer
def load_mag_model_and_tokenizer(model_args):
    checkpoint = model_args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, use_fast=False)
    config = InternVLChatConfig.from_pretrained(checkpoint)
    #--from_pretrained 已经正确加载模型权重了。保险起见再用safetensor加载一遍
    model = InternVLMagnifier.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, config=config, 
            llm_tokenizer=tokenizer, args=model_args)
    #加载权重到模型
    if not hasattr(config, "seg_model"):
        model.init_seg_mformer()
    # 遍历分片文件并加载权重
    state_dict = {}
    for i in range(1, 6):  # 根据分片文件数量动态调整
        shard_path = f"{model_args.checkpoint}/model-0000{i}-of-00005.safetensors"
        shard_state_dict = load_file(shard_path)
        state_dict.update(shard_state_dict)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(dtype=torch.bfloat16).cuda()
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    return model, tokenizer

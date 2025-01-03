# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from typing import List, Optional, Tuple, Union

import torch.distributed as dist
import torch.utils.checkpoint
import transformers
from internvl.conversation import get_conv_template
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.phi3.modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig
from internvl.model.internvl_chat.modeling_intern_vit import InternVisionModel, has_flash_attn
from .Qformer import BertConfig, BertLMHeadModel
from .Mformer import TwoWayTransformer
import os
from transformers import AutoTokenizer
from internvl.model.internvl_seg import (
    SegInternVLForCausalLM,
)
from internvl.train.constants import SEG_TOKEN, IMG_CONTEXT_TOKEN
from safetensors.torch import load_file


logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))
class Mformer_config:
    def __init__(self, args):
        num_layers = args.num_layers,
        num_heads = args.num_heads,
        max_length = args.max_length,
        embedding_dim = args.embedding_dim,


class InternVLMagnifier(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer']
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True, args=None, llm_tokenizer=None):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]
        # Enable Flash Attention if supported, otherwise fall back to eager attention.
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config.attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        self.vit_hidden_size = config.vision_config.hidden_size
        self.llm_hidden_size = config.llm_config.hidden_size
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size


        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

        self.mix_mlp = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
        self.seg_mlp = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
        self.config = config
        self.llm_tokenizer = llm_tokenizer
        if not hasattr(self.config, "seg_model"):
            #运行这段代码
            self.config.segmentation_model_path = args.segmentation_model_path
            self.config.seg_model = args.seg_model
            self.config.mixed_strategy = args.mixed_strategy
            self.config.Mformer = dict(
                num_layers = args.num_layers,
                num_heads = args.num_heads,
                max_length = args.max_length,
                embedding_dim = args.embedding_dim,
            )
            self.config.seg_tokens = args.seg_tokens
        else:
            self.init_seg_mformer() 
    #--------------初始化相关组件
    def init_seg_mformer(self):
        logger.info("Initializing SEG Modules...")
        self.seg_model, self.seg_tokenizer = self.init_seg_model(self.config)
        self.init_neck()
        self.Mformer = self.init_Mformer(self.config.Mformer)
        self.max_length = self.config.Mformer['max_length']
        self.mixed_strategy = self.config.mixed_strategy

    def init_Mformer(self, kwargs):
        embedding_dim = kwargs['embedding_dim']
        Mformer = TwoWayTransformer(
                depth=kwargs['num_layers'],
                embedding_dim=embedding_dim,
                mlp_dim=embedding_dim*2,
                num_heads=kwargs['num_heads'],
            )
        return Mformer

    def init_seg_model(self, config):
        # work_dirs/InternVL2_5-1B_seg_8w_color
        tokenizer = AutoTokenizer.from_pretrained(config.seg_model, trust_remote_code=True, use_fast=False)
        seg_config = InternVLChatConfig.from_pretrained(config.seg_model)
        seg_model_args = {
            "seg_token_idx": tokenizer(SEG_TOKEN, add_special_tokens=False).input_ids[0],
            "segmentation_model_path": config.segmentation_model_path,
            "tokenizer": tokenizer,
        }
        #--from_pretrained 已经正确加载模型权重了。保险起见再用safetensor加载一遍
        model = SegInternVLForCausalLM.from_pretrained(
            config.seg_model, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,config=seg_config, **seg_model_args)
        file_path = os.path.join(config.seg_model, 'model.safetensors')
        state_dict = load_file(file_path)
        # 加载权重到模型
        model.load_state_dict(state_dict, strict=False)
        model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        return model, tokenizer
    
    def init_neck(self):
        self.neck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False),
        )
    #--------------初始化相关组件
        
    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.llm_arch_name == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()
    def get_mixed_embs(self, mixed_img_emb, upscale_seg_img):
        def mix_two_emb(mixed_strategy, seg_emb, mixed_emb):
            if mixed_strategy == 'only_seg' or mixed_strategy == 'only_seg_q':
                return seg_emb
            elif mixed_strategy == 'only_mix':
                return mixed_emb
            elif mixed_strategy == 'concat_seq':
                return torch.cat((mixed_emb, seg_emb), dim=1)
            elif mixed_strategy == 'interleave_seq':
                # 交错拼接
                bs, num_tokens, dim = mixed_emb.shape
                interleaved = torch.stack((mixed_emb, seg_emb), dim=2)  # [bs, num_tokens, 2, dim]
                interleaved = interleaved.reshape(bs, num_tokens * 2, dim)  # [bs, num_tokens*2, dim]
                return interleaved
            elif mixed_strategy == 'no_mix_concat':
                #TODO
                return torch.cat((mixed_emb, seg_emb), dim=1)
            elif mixed_strategy == 'concat_dim':
                #TODO
                pass
        assert mixed_img_emb.shape[1] == upscale_seg_img.shape[1]
        seg_emb = self.seg_mlp(upscale_seg_img)
        mixed_emb = self.mix_mlp(mixed_img_emb)
        return mix_two_emb(self.mixed_strategy, seg_emb, mixed_emb)

    def get_final_input_embs(self, vlm, seg):
        # with torch.no_grad():
        bs = vlm['input_ids'].shape[0]
        #---得到seg的图像编码，并放缩到internvit维度。upscale_seg_img
        image_seg_embs = self.seg_model.generate(**seg) #shape:[bs, dim, h, w] [bs, 256, 64, 64]
        upscale_seg_img = self.neck(image_seg_embs) #shape:[bs, 1024, 16, 16]
        upscale_seg_img = upscale_seg_img.permute(0, 2, 3, 1).contiguous() #shape:[bs, 16, 16, 1024]
        # upscale_seg_img = upscale_seg_img.flatten(2).permute(0, 2, 1) #shape[bs, 256, 1024]
        upscale_seg_img = self.pixel_shuffle(upscale_seg_img, scale_factor=self.downsample_ratio) #shape:[bs, 8,8,4096]
        upscale_seg_img = upscale_seg_img.reshape(upscale_seg_img.shape[0], -1, upscale_seg_img.shape[-1]) #shape:[bs, 64, 4096]
        # assert (vlm['input_ids'] == self.img_context_token_id).sum().item() == bs * upscale_seg_img.shape[1]
        #----得到Internvit的visual token：vlm_img
        pixel_values = vlm['pixel_values'] #shape:[bs* patches, 3, 448, 448] 其中每个样本的patches可能不一样 
        vit_embeds = self.extract_feature(pixel_values) #shape:[n, 256, 4096] n为pacth的累积和
        vit_embeds = vit_embeds[vlm['image_flags'] == 1]
        vlm_img = torch.zeros(bs, self.max_length, vit_embeds.shape[-1]).to(device=vit_embeds.device, dtype=vit_embeds.dtype) #shape:[bs, max_length, 4096]
        seg_attn_vlm_mask = torch.zeros(bs, self.max_length).to(device=vit_embeds.device, dtype=torch.bool) #shape:[bs, max_length]
        img_position_ids = torch.zeros(bs, self.max_length).to(device=vit_embeds.device, dtype=torch.long) #shape:[bs, max_length]
        offset = vlm['offset']
        vit_tokens = vit_embeds.shape[1]
        assert offset[-1] == vit_embeds.shape[0]
        for i in range((len(offset) - 1)):
            start_i, end_i = offset[i], offset[i + 1] #确定每张图片切分了几个patch
            length = (end_i - start_i) * vit_tokens
            vlm_img[i][:length] = vit_embeds[start_i: end_i].reshape(-1, vit_embeds.shape[-1])
            seg_attn_vlm_mask[i][:length] = 1
            img_position_ids[i] = seg_attn_vlm_mask[i].long().cumsum(-1) - 1
            img_position_ids[i][length:] = 0   #padding 位置取0
        #---使用M_former做cross-attn
        mixed_img_emb, keys = self.Mformer(
            vlm_img,
            img_position_ids,
            upscale_seg_img,
            seg_attn_vlm_mask
        )
        #--使用img emb 替换掉input emb中的img部分
        #TODO：序列拼接 维度拼接 直接替换
        mixed_embs = self.get_mixed_embs(
            mixed_img_emb,
            upscale_seg_img,
        )
        #---暂时用拼接策略
        assert (vlm['input_ids'] == self.img_context_token_id).sum().item() == bs * mixed_embs.shape[1]
        #---
        input_ids = vlm['input_ids']
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
        B, N, C = input_embeds.shape
        size = [B, N, C]
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + mixed_embs.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            from IPython import embed; embed(); exit()
            vit_embeds = mixed_embs.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={mixed_embs.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + mixed_embs[:n_token]
            ignore_flag = True
        return input_embeds, size, ignore_flag
    def forward(self, vlm, seg):
        input_embeds, size, ignore_flag = self.get_final_input_embs(vlm ,seg)
        outputs = self.vlm_forward(input_embeds=input_embeds, size=size, ignore_flag=ignore_flag, **vlm)
        return outputs
    def vlm_forward(
            self,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            loss_weight: Optional[List] = None,
            loss_reduction_all_gather: Optional[bool] = False,
            input_embeds: Optional[torch.FloatTensor] = None,
            size: Optional[int] = None,
            ignore_flag=None,
            **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        B, N, C = size
        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None and loss_weight is not None:
            loss_weight = torch.tensor(loss_weight, dtype=torch.float32, device=labels.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weight[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_weights = shift_weights.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            shift_weights = shift_weights.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            shift_weights_sum = shift_weights.sum()
            if loss_reduction_all_gather:
                dist.all_reduce(shift_weights_sum, op=dist.ReduceOp.AVG)

            loss = loss * shift_weights
            loss = loss.sum() / shift_weights_sum
            if ignore_flag:
                loss = loss * 0.0
        elif labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            #走这里
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1) #shape[N, 32, 32, 1024]
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1]) #shape[N, 256, 4096]
        # vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def batch_chat(self, vlm, seg, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None, tokenizer=None,
                   **kwargs):
        # if history is not None or return_history:
        #     print('Now multi-turn chat is not supported in batch_chat.')
        #     raise NotImplementedError

        # if image_counts is not None:
        #     num_patches_list = image_counts
        #     print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        # img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        # self.img_context_token_id = img_context_token_id

        # if verbose and pixel_values is not None:
        #     image_bs = pixel_values.shape[0]
        #     print(f'dynamic ViT batch size: {image_bs}')

        # queries = []
        # for idx, num_patches in enumerate(num_patches_list):
        #     question = questions[idx]
        #     if pixel_values is not None and '<image>' not in question:
        #         question = '<image>\n' + question
        #     template = get_conv_template(self.template)
        #     template.system_message = self.system_message
        #     template.append_message(template.roles[0], question)
        #     template.append_message(template.roles[1], None)
        #     query = template.get_prompt()

        #     image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
        #     query = query.replace('<image>', image_tokens, 1)
        #     queries.append(query)

        # tokenizer.padding_side = 'left'
        # model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # input_ids = model_inputs['input_ids'].to(device)
        # attention_mask = model_inputs['attention_mask'].to(device)
        template = get_conv_template(self.template)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            vlm=vlm,
            seg=seg,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(self, vlm, seg, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False, tokenizer=None, **kwargs):
        # if history is None and pixel_values is not None and '<image>' not in question:
        #     question = '<image>\n' + question

        # if num_patches_list is None:
        #     num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        # assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        # img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        # self.img_context_token_id = img_context_token_id

        # template.system_message = self.system_message

        # history = [] if history is None else history
        # for (old_question, old_answer) in history:
        #     template.append_message(template.roles[0], old_question)
        #     template.append_message(template.roles[1], old_answer)
        # template.append_message(template.roles[0], question)
        # template.append_message(template.roles[1], None)
        # query = template.get_prompt()

        # if verbose and pixel_values is not None:
        #     image_bs = pixel_values.shape[0]
        #     print(f'dynamic ViT batch size: {image_bs}')

        # for num_patches in num_patches_list:
        #     image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
        #     query = query.replace('<image>', image_tokens, 1)

        # model_inputs = tokenizer(query, return_tensors='pt')
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # input_ids = model_inputs['input_ids'].to(device)
        # attention_mask = model_inputs['attention_mask'].to(device)



        template = get_conv_template(self.template)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            vlm=vlm,
            seg=seg,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        return response
        # history.append((question, response))
        # if return_history:
        #     return response, history
        # else:
        #     query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
        #     query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
        #     if verbose:
        #         print(query_to_print, response)
        #     return response

    @torch.no_grad()
    def generate(
            self,
            vlm, 
            seg,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        input_embeds, size, ignore_flag = self.get_final_input_embs(vlm ,seg)
        attention_mask = vlm['attention_mask']
        B, N, C = size
        input_embeds = input_embeds.reshape(B, N, C)


        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )
        return outputs

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

if __name__ == "__main__":
    pass
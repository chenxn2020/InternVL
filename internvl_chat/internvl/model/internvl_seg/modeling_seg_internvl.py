# --------------------------------------------------------
# LISA: Reasoning Segmentation via Large Language Model
# Licensed under Apache-2.0 license [see LICENSE for details]
# Authors: Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, Jiaya Jia
# --------------------------------------------------------
# GSVA: Generalized Segmentation via Multimodal Large Language Models
# Modified by Zhuofan Xia
# --------------------------------------------------------

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Optional, Tuple, Union
from .losses import dice_loss, sigmoid_ce_loss
from transformers.modeling_utils import PreTrainedModel
from internvl.model.segment_anything import build_sam_vit_h, build_sam_vit_l, build_sam_vit_b
from internvl.model.internvl_chat import (
    InternVisionConfig,
    InternVisionModel,
    InternVLChatConfig,
    InternVLChatModel,
)

class SAMMetaModel(nn.Module):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            #运行这段代码
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.segmentation_model_path = kwargs.get("segmentation_model_path", None)
        else:
            self.segmentation_model_path = kwargs.get("segmentation_model_path", None)
            self.init_seg_and_proj(self.config)

    def init_seg_and_proj(self, config):
        #类初始化的时候不运行这段代码！！！,在main中单独运行
        # SAM
        print("Loading SAM")
        builder_sam = build_sam_vit_h if "sam_vit_h" in self.segmentation_model_path else \
            build_sam_vit_l if "sam_vit_l" in self.segmentation_model_path else build_sam_vit_b
        self.visual_model = builder_sam(self.segmentation_model_path)
        # Projection layer for SAM
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])

class SamModel(SAMMetaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False
        self.seg_token_idx = kwargs.get("seg_token_idx", 0)

class SegInternVLForCausalLM(InternVLChatModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        if not hasattr(config, "train_mask_decoder"):
            #会运行这段代码
            #--这里修改了model config。所以必须制定vision tower的本地路径
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.llm_tokenizer = kwargs.get("tokenizer", None)
        #--internvl model
        #---sam model
        self.model = SamModel(config, seg_token_idx=self.seg_token_idx, **kwargs)
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        #TODO：得看下internvl里的patches个数
        self.post_init()
    
    def get_model(self):
        return self.model

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(
        self,
        #---InterChatModel input
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        statistics: Optional[torch.LongTensor] = None,
        loss_weight: Optional[List] = None,
        loss_reduction_all_gather: Optional[bool] = False,
        #---SAM input
        images: torch.FloatTensor = None,
        resize_list: List[tuple] = None,
        do_segs: List[bool] = None,
        masks_list: List[torch.FloatTensor] = None,
        label_list: List[torch.Tensor] = None,
        offset: torch.LongTensor = None,
        inference: bool = False,
        reeval: bool = False,
        conversations: Optional[List] = None,
        **kwargs,
    ):
        device, dtype = images.device, images.dtype
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1
        
        # # ---如果是多轮对话就需要对图片进行expand
        # images_clip_list = []
        # for i in range(len(offset) - 1): # offset marks each begin and end index for each images.
        #     start_i, end_i = offset[i], offset[i + 1]
        #     images_clip_i = (pixel_values[i].unsqueeze(0).expand(end_i - start_i, -1, -1, -1).contiguous())
        #     images_clip_list.append(images_clip_i)
        # images_clip = torch.cat(images_clip_list, dim=0)
        #----

        #-----
        # VLM inference, obtain InternVL output
        images_clip = pixel_values
        output = super().forward(
            pixel_values=images_clip,
            attention_mask=attention_mask,
            input_ids=input_ids, #input_ids 就是整个prompt模版下的问答
            labels=labels,
            image_flags=image_flags,
            position_ids=position_ids,
            output_hidden_states=True
        )
        output_hidden_states = output.hidden_states #所有层的embeddings,tuple形式
        hidden_states = []
        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

        seg_token_mask = input_ids == self.seg_token_idx     
        #--这部分代码是说如果一个样本有多个SEG token会导致pred SEG不知道是哪条数据的预测SEG,所以要将pre_emb分组到各自的样本里 
        pred_embeddings = last_hidden_state[seg_token_mask] #shape[SEG_num, hidden_dim]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
        seg_token_offset = seg_token_counts.cumsum(-1) #累积和
        seg_token_offset = torch.cat(
            [torch.tensor([0], dtype=torch.int64, device=device), seg_token_offset], dim=0
        )     
        pred_embeddings_ = []
        num_pred_embs = len(seg_token_offset) - 1 #bs
        for i in range(num_pred_embs):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_ #以列表形式分组保存每条数据的预测SEG
        #---
        pred_masks = []
        pred_ious = []
        src_images = []
        mask_img_map = [(t >= offset).long().argmin().item() - 1 for t in range(num_pred_embs)]
        for i in range(len(pred_embeddings)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(dtype)
            low_res_masks, iou_predictions, src_image = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[mask_img_map[i]].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[mask_img_map[i]],
                original_size=label_list[mask_img_map[i]].shape
            )
            pred_masks.append(pred_mask[:, 0])
            pred_ious.append(iou_predictions[:, 0])
            src_images.append(src_image)
        # src_images = torch.cat(src_images, dim=0)
        
        model_output = output
        gt_masks = masks_list

        for b in range(batch_size):
            for pm, gm in zip(pred_masks[b], gt_masks[b]):
                assert pm.shape == gm.shape, f"b_idx: {b}, pm.shape: {pm.shape}, gm.shape: {gm.shape}"
        #--如果inference
        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }
        from IPython import embed; embed(); exit()
        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        loss = 0
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            if batch_idx >= len(gt_masks):
                raise ValueError(f"gt_masks are not in good shape with b_idx={batch_idx} >= len(gt_masks)={len(gt_masks)}, also len(preds)={len(pred_masks)}.")
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]
            if (
                gt_mask.shape[0] != pred_mask.shape[0]
            ):
                i0, i1 = input_ids[0], input_ids[1]
                # i0, i1 = i0[i0 != IMAGE_TOKEN_INDEX], i1[i1 != IMAGE_TOKEN_INDEX]
                print(f"gt: {gt_mask.shape}, pred: {pred_mask.shape}\n" + \
                    f"Prompt0: {self.llm_tokenizer.decode(i0, skip_special_tokens=True)}\n" + \
                    f"Prompt1: {self.llm_tokenizer.decode(i1, skip_special_tokens=True)}\n" + \
                    f"GT_MASK sum :{gt_mask.sum(dim=(1, 2))}\n"
                )
                raise RuntimeError("Found it!")
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            ) #二分类损失
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]
        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss
        loss = ce_loss + mask_loss
        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }
    @torch.no_grad()
    def generate(self,
        #---InterChatModel input
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        statistics: Optional[torch.LongTensor] = None,
        loss_weight: Optional[List] = None,
        loss_reduction_all_gather: Optional[bool] = False,
        #---SAM input
        images: torch.FloatTensor = None,
        offset: torch.LongTensor = None,
        **kwargs,
    ):
        device, dtype = images.device, images.dtype
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1
        
        #---如果是多轮对话就需要对图片进行expand
        # images_clip_list = []
        # for i in range(len(offset) - 1): # offset marks each begin and end index for each images.
        #     start_i, end_i = offset[i], offset[i + 1]
        #     images_clip_i = (pixel_values[i].unsqueeze(0).expand(end_i - start_i, -1, -1, -1).contiguous())
        #     images_clip_list.append(images_clip_i)
        # images_clip = torch.cat(images_clip_list, dim=0)
        images_clip = pixel_values
        #-----
        # VLM inference, obtain InternVL output
        output = super().forward(
            pixel_values=images_clip,
            attention_mask=attention_mask,
            input_ids=input_ids, #input_ids 就是整个prompt模版下的问答
            labels=labels,
            image_flags=image_flags,
            position_ids=position_ids,
            output_hidden_states=True
        )
        output_hidden_states = output.hidden_states #所有层的embeddings,tuple形式
        hidden_states = []
        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

        seg_token_mask = input_ids == self.seg_token_idx     
        #--这部分代码是说如果一个样本有多个SEG token会导致pred SEG不知道是哪条数据的预测SEG,所以要将pre_emb分组到各自的样本里 
        pred_embeddings = last_hidden_state[seg_token_mask] #shape[SEG_num, hidden_dim]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
        seg_token_offset = seg_token_counts.cumsum(-1) #累积和
        seg_token_offset = torch.cat(
            [torch.tensor([0], dtype=torch.int64, device=device), seg_token_offset], dim=0
        )     
        pred_embeddings_ = []
        num_pred_embs = len(seg_token_offset) - 1 #bs
        for i in range(num_pred_embs):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_ #以列表形式分组保存每条数据的预测SEG
        #---
        src_images = []
        mask_img_map = [(t >= offset).long().argmin().item() - 1 for t in range(num_pred_embs)]
        for i in range(len(pred_embeddings)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(dtype)
            low_res_masks, iou_predictions, src_image = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[mask_img_map[i]].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            src_images.append(src_image)
        src_images = torch.cat(src_images, dim=0)
        return src_images

class Test(InternVLChatModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        
if __name__ == "__main__":
    model_name_or_path = '/cpfs01/user/caixinyu/lisiqi/InternVL/internvl_chat/pretrained/InternVL2_5-1B'
    config = InternVLChatConfig.from_pretrained(model_name_or_path)
    model = Test.from_pretrained(
        model_name_or_path, torch_dtype=torch.bfloat16, config=config)
    from IPython import embed; embed(); exit()
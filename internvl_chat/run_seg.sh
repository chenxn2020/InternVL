set -x

GPUS=${GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-8}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
NUM_WORKERS=12


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=32120
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export SETUPTOOLS_USE_DISTUTILS=local

OUTPUT_DIR='work_dirs/InternVL2_5-1B_seg_8w_color_sam'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 8
# batch size per gpu: 4
# gradient accumulation steps: 4
# total batch size: 128 
# epoch: 1
# CUDA_VISIBLE_DEVICES=7 torchrun \
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_seg.py \
  --model_name_or_path "/cpfs01/user/caixinyu/lisiqi/InternVL/internvl_chat/pretrained/InternVL2_5-1B" \
  --conv_style "internvl2_5_seg" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "/cpfs01/user/caixinyu/chenxiangnan/dsw/InternVL_SEG/internvl_chat/shell/data/color_train.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 12 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --dataloader_num_workers $NUM_WORKERS \
  --bf16 True \
  --num_train_epochs 12 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 2 \
  --learning_rate 4.5e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8000\
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  --segmentation_model_path "/cpfs01/shared/ADLab/hug_ckpts/chenxiangnan/sam_vit_h_4b8939.pth" \
  --do_seg \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
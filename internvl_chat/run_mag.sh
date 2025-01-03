set -x

GPUS=${GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=30129
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
#mix :[concat_seq, only_seg, only_mix, interleave_seq, concat_dim]
MIX="only_seg_q"
OUTPUT_DIR='work_dirs/internvl_magnifier/internvl2_5_8b'
OUTPUT_DIR="${OUTPUT_DIR}/${MIX}"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 8
# batch size per gpu: 4
# gradient accumulation steps: 4
# total batch size: 128
# epoch: 1
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  ./internvl/train/internvl_mag.py \
  --mixed_strategy $MIX \
  --seg_tokens 64 \
  --learning_rate 5e-4 \
  --model_name_or_path "/cpfs01/user/caixinyu/chenxiangnan/dsw/shared/models/chenxiangnan/InternVL2_5-8B" \
  --seg_model "/cpfs01/user/caixinyu/chenxiangnan/dsw/InternVL_SEG/internvl_chat/work_dirs/InternVL2_5-1B_seg_8w_color" \
  --conv_style "internvl2_5_mag" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data/color_train.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 12 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone True \
  --freeze_seg True \
  --vision_select_layer -1 \
  --dataloader_num_workers 0 \
  --bf16 True \
  --num_train_epochs 8 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 2 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --remove_unused_columns False \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

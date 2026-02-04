set -x

#export CUDA_VISIBLE_DEVICES="0,1,2,3"
GPUS=${GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-512}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
NPROC_PER_NODE=${NPROC_PER_NODE:-$GPUS}

META_PATH=${META_PATH:-"shell/data/meta_visualprm400k.json"}
MODEL_PATH=${MODEL_PATH:-"OpenGVLab/InternVL2_5-8B"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"configs/zero_stage3_config.json"}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
MASTER_PORT=${MASTER_PORT:-4322}
export MASTER_PORT
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
OUTPUT_DIR=${OUTPUT_DIR:-"work_dirs/internvl_chat_v2_5/visualprm400k_full_hard"}

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

python -m torch.distributed.run \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --nproc_per_node=${NPROC_PER_NODE} \
  --master_port=${MASTER_PORT} \
  src/internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "${MODEL_PATH}" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "${META_PATH}" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.4 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --save_strategy "steps" \
  --save_only_model True \
  --save_steps 50 \
  --save_total_limit 18 \
  --learning_rate 1e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

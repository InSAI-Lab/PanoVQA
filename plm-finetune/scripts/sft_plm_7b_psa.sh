
#!/bin/bash

# Distributed training configuration
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=/home/hk-project-pai00053/wd1434/workspace/huggingface/Qwen2.5-VL-7B-Instruct  # Using HuggingFace model ID

# Training hyperparameters
lr=5e-6
batch_size=8
grad_accum_steps=4

# Training entry point
entry_file=plm/train/train_qwen_attAdoptor.py

# Dataset configuration (replace with public dataset names)
datasets=NuScenes_mini_train,DeepAccident_mini_train

bottle=128
sparse_k=512
index_heads=4
index_dim=16
max_relative_position=128

# Output configuration
run_name="plm_sparse_attention"
output_dir="./output/plm/plm7b_psa_fast_bs=64(adapter,llm,mlp)"

# output_dir=./output/output_finetune(vit,mlp)_first_phrase
log_dir=${output_dir}/logs

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --logging_dir ${log_dir} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --tune_mm_adaptorformer True \
    --adaptor_name "panorama_sparse_attention" \
    --bottle_dim ${bottle} \
    --sparse_k ${sparse_k} \
    --index_heads ${index_heads} \
    --index_dim ${index_dim} \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing False \
    --dataloader_num_workers 8 \
    --run_name ${run_name} \
    --report_to wandb"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}

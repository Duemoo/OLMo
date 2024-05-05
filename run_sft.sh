#!/bin/bash

scratch_dir=/home/jinho/repos/OLMo
export SCRATCH_DIR=${scratch_dir}
export CUDA_VISIBLE_DEVICES=1


# Choose between [40000, 113000, 339000]
model_size=1b
ckpt_step=113000
dataset_path="[${scratch_dir}/fictional_knowledge/fictional_knowledge_paraphrased.json]"
# ckpt_dir=/mnt/nas/jinho/olmo_checkpoints/${model_size}
ckpt_dir=/home/hoyeon/official_checkpoints/${model_size}

torchrun --nproc_per_node=1 scripts/train.py configs/official/OLMo-${model_size}-sft.yaml \
    --run_name="OLMo-${model_size}_sft_step${ckpt_step}" \
    --data.paths=${dataset_path} \
    --load_path=${ckpt_dir}/step${ckpt_step}-unsharded \
    --device_eval_batch_size=8 \
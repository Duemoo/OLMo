#!/bin/bash

#SBATCH --job-name=OLMo-1B_sft_step113000        # Job name
#SBATCH -o out_%j.txt                            # Path to output log file (%j expands to job name)
#SBATCH -e err_&j.err                            # Path to error log file (%j expands to job name)
#SBATCH --partition=LocalQ                       # Partition name
#SBATCH --nodes=1                                # Request one node
#SBATCH --ntasks=1                               # Request one task (default)
#SBATCH --cpus-per-task=1                        # Number of CPU cores per task
#SBATCH --time=24:00:00                          # Time limit
#SBATCH --gres=gpu:2                             # Number of GPUs to be allocated

scratch_dir=/home/jinho/repos/OLMo
export SCRATCH_DIR=${scratch_dir}

model_size=1B
# Choose between [40000, 113000, 339000]
ckpt_step=113000
dataset_path="[${scratch_dir}/fictional_knowledge/fictional_knowledge_paraphrased.json]"
# ckpt_dir=/mnt/nas/jinho/olmo_checkpoints/${model_size}
ckpt_dir=/home/hoyeon/official_checkpoints/${model_size}

srun torchrun --nproc_per_node=2 scripts/train.py configs/official/OLMo-${model_size}-sft.yaml \
    --run_name="OLMo-${model_size}_sft_step${ckpt_step}" \
    --data.paths=${dataset_path} \
    --load_path=${ckpt_dir}/step${ckpt_step}-unsharded \
    --device_eval_batch_size=8 \
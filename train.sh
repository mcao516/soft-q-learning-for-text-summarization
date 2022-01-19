#!/bin/bash
# Load python enviroment
if [ ${HOSTNAME:0:5} = "login" ] || [ ${HOSTNAME:0:2} = "cn" ]; then
    echo "Load enviroment on MILA cluster"
    module load cuda/11.0
    module load python/3.8
    source $HOME/envABERT/bin/activate
else
    echo "Load enviroment on CC"
    module load StdEnv/2020 gcc/9.3.0 cuda/11.0
    module load arrow/5.0.0
    module load python/3.8

    if [ ${HOSTNAME:0:3} = "blg" ]; then
        source $SCRATCH/envABERT/bin/activate
    elif [ ${HOSTNAME:0:2} = "ng" ] || [ ${HOSTNAME:0:3} = "cdr" ] || [ ${HOSTNAME:0:2} = "na" ]; then
        source $HOME/envABERT/bin/activate
    else
        echo "Unknown cluster!!!"
    fi
fi


MODEL_NAME_OR_PATH=$SCRATCH/huggingface/bart-base
OUTPUT_DIR=$SCRATCH/BART_base_xsum_soft_q_50sampling

CUDA_LAUNCH_BLOCKING=1
accelerate launch --config_file gpu_3_config.yaml run_summarization_no_trainer.py \
    --job_name bart-base-50sampling \
    --project_name summarization-soft-q \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name xsum \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --preprocessing_num_workers 16 \
    --num_warmup_steps 1000 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 1 \
    --num_beams 6 \
    --num_train_epochs 6 \
    --output_dir $OUTPUT_DIR;
    # --source_prefix "summarize: " \
    # --dataset_config "3.0.0" \
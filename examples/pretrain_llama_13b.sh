#! /bin/bash

# This is the example script to pretrain a 7B LLaMA model on a TPU v4-512 pod.
# These hyperparameters are the ones we used to train the OpenLLaMA 7B model on
# the RedPajama dataset. To use this on TPU pod, you need to run this
# script on every hosts in a TPU pod.

# Put your WANDB API key here to enable logging to wandb.
export WANDB_API_KEY='99c1cfcf5ab402b2d7df6da383d1645fe6da06b6'

# TPU specific flags to improve training throughput
# export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'


python -m EasyLM.models.llama.llama_train \
    --mesh_dim='1,2,4' \
    --dtype='bf16' \
    --total_steps=250000 \
    --log_freq=50 \
    --save_model_freq=0 \
    --save_milestone_freq=2500 \
    --load_llama_config='13b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='params::/mnt/data/EasyLM/model/easylm/Llama-2-13b-hf' \
    --tokenizer.vocab_file='/mnt/data/EasyLM/model/Llama-2-13b-hf/tokenizer.model' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.1 \
    --optimizer.adamw_optimizer.lr=3e-4 \
    --optimizer.adamw_optimizer.end_lr=3e-5 \
    --optimizer.adamw_optimizer.lr_warmup_steps=2000 \
    --optimizer.adamw_optimizer.lr_decay_steps=250000 \
    --train_dataset.type='huggingface' \
    --num_epochs=1 \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.huggingface_dataset.path='datablations/c4-subsets' \
    --train_dataset.huggingface_dataset.split='validation' \
    --train_dataset.huggingface_dataset.seq_length=2048 \
    --train_dataset.huggingface_dataset.batch_size=1 \
    --checkpointer.save_optimizer_state=True \
    --logger.online=True \
    --log_freq 1000 \
    --logger.prefix='EasyLM' \
    --logger.project="open_llama_13b" \
    --logger.output_dir="/mnt/data/EasyLM/output" \
    --logger.wandb_dir="$HOME/experiment_output/open_llama_13b" \
|& tee $HOME/output.txt

#     # --train_dataset.text_processor.fields='text' \
    # --train_dataset.json_dataset.path='/path/to/shuffled/redpajama/dataset' \
#! /bin/bash

# This is the example script to pretrain a 7B LLaMA model on a TPU v4-512 pod.
# These hyperparameters are the ones we used to train the OpenLLaMA 7B model on
# the RedPajama dataset. To use this on TPU pod, you need to run this
# script on every hosts in a TPU pod.

# Put your WANDB API key here to enable logging to wandb.
export WANDB_API_KEY='99c1cfcf5ab402b2d7df6da383d1645fe6da06b6'

git config --global credential.helper store

huggingface-cli login --token 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

# TPU specific flags to improve training throughput
export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'


python -m EasyLM.models.llama.llama_train \
    --mesh_dim='1,2,-1' \
    --dtype='bf16' \
    --initialize_jax_distributed=True \
    --total_steps=250000 \
    --log_freq=256 \
    --save_model_freq=0 \
    --save_milestone_freq=4096 \
    --load_llama_config='7b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='params::gs://data-selection-bucket/easylm/Llama-2-7b-chat-hf' \
    --tokenizer.vocab_file='gs://data-selection-bucket/Llama-2-7b-chat-hf/tokenizer.model' \
    --optimizer.type='adamw' \
    --optimizer.accumulate_gradient_steps=8 \
    --optimizer.adamw_optimizer.weight_decay=0.00 \
    --optimizer.adamw_optimizer.lr=2e-5 \
    --optimizer.adamw_optimizer.end_lr=5e-6 \
    --optimizer.adamw_optimizer.warmup_ratio=0.03 \
    --num_epochs=3 \
    --train_dataset.text_processor.fields='[question+prompt],answer' \
    --train_dataset.type='classification_json_torch' \
    --train_dataset.json_torch_dataset.path='gs://data-selection-bucket/data/processed/wos/wos_train.json' \
    --train_dataset.json_torch_dataset.seq_length=2048 \
    --train_dataset.json_torch_dataset.batch_size=8 \
    --train_dataset.json_torch_dataset.num_workers=24 \
    --checkpointer.save_optimizer_state=True \
    --llama.scan_attention=True \
    --llama.scan_mlp=True \
    --logger.online=True \
    --logger.prefix='EasyLM' \
    --logger.project="open_llama_7b" \
    --logger.output_dir="gs://data-selection-bucket/easylm/output" \
    --logger.wandb_dir="$HOME/experiment_output/open_llama_7b" \
    --log_all_worker=False
| & tee $HOME/output.txt

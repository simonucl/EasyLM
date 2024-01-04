MODEL_PATH='gs://data-selection-bucket/easylm/output/eb4c2882129e4fc1a73a4dbafbbcef2f/streaming_train_state_1920'
# EPOCH='epoch_3'

python -m EasyLM.models.llama.convert_easylm_to_hf \
    --load_checkpoint="params::$MODEL_PATH" \
    --tokenizer_path="/mnt/data/Llama-2-7b-hf/tokenizer.model" \
    --model_size='7b' \
    --output_dir="/mnt/data/EasyLM/model/Llama-2-7b-hf-lima" \
    --hf_dir="simonycl/llama-2-7b-hf-sharegpt-full-ft-lima" \

cd /mnt/data/lm-evaluation-harness
bash eval_model.sh /mnt/data/EasyLM/model/Llama-2-7b-hf-lima Llama-2-7b-hf-lima
# bash eval_model.sh simonycl/llama-2-7b-hf-sharegpt-full-ft-$EPOCH Llama-2-7b-hf-sharegpt-$EPOCH 

cd /mnt/data/data-selection/

bash scripts/eval/cohere.sh /mnt/data/EasyLM/model/Llama-2-7b-hf-lima Llama-2-7b-hf-sharegpt-lima

###############################################################
# cd /mnt/data/EasyLM/examples

# MODEL_PATH='gs://data-selection-bucket/easylm/output/3b8f6401b83c41f4bc8462a95c460e1d/streaming_train_state_12288'
# EPOCH='epoch_2'

# python -m EasyLM.models.llama.convert_easylm_to_hf \
#     --load_checkpoint="params::$MODEL_PATH" \
#     --tokenizer_path="/mnt/data/Llama-2-7b-hf/tokenizer.model" \
#     --model_size='7b' \
#     --output_dir="/mnt/data/EasyLM/model/Llama-2-7b-hf-sharegpt-$EPOCH" \
#     --hf_dir="simonycl/llama-2-7b-hf-sharegpt-full-ft-$EPOCH" \

# cd /mnt/data/lm-evaluation-harness
# bash eval_model.sh /mnt/data/EasyLM/model/Llama-2-7b-hf-sharegpt-$EPOCH Llama-2-7b-hf-sharegpt-$EPOCH 

# cd /mnt/data/data-selection/

# bash scripts/eval/cohere.sh /mnt/data/EasyLM/model/Llama-2-7b-hf-sharegpt-$EPOCH Llama-2-7b-hf-sharegpt-$EPOCH

# # nohup bash examples/convert_easylm_hf.sh > examples/convert_easylm_hf.log 2>&1 &

# # nohup bash scripts/eval/cohere.sh /mnt/data/EasyLM/model/Llama-2-7b-hf-sharegpt-epoch_3 Llama-2-7b-hf-sharegpt-epoch_3 > logs/eval_mmlu_sharegpt_3epoch.log 2>&1 &
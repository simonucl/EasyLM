MODEL_PATH='gs://data-selection-bucket/easylm/output/f605cd2481b943819dfe1f3eac1142d9/streaming_train_state_29928' 
EPOCH='epoch_3'

python -m EasyLM.models.llama.convert_easylm_to_hf \
    --load_checkpoint="params::$MODEL_PATH" \
    --tokenizer_path="/mnt/data/Llama-2-7b-hf/tokenizer.model" \
    --model_size='7b' \
    --output_dir="/mnt/data/EasyLM/model/Llama-2-7b-hf-sharegpt-$EPOCH" \
    --hf_dir='simonycl/llama-2-7b-hf-sharegpt-full-ft-$EPOCH' \

MODEL_PATH='gs://data-selection-bucket/easylm/output/3b8f6401b83c41f4bc8462a95c460e1d/streaming_train_state_12288'
EPOCH='epoch_2'

python -m EasyLM.models.llama.convert_easylm_to_hf \
    --load_checkpoint="params::$MODEL_PATH" \
    --tokenizer_path="/mnt/data/Llama-2-7b-hf/tokenizer.model" \
    --model_size='7b' \
    --output_dir="/mnt/data/EasyLM/model/Llama-2-7b-hf-sharegpt-$EPOCH" \
    --hf_dir='simonycl/llama-2-7b-hf-sharegpt-full-ft-$EPOCH' \
    
# nohup bash examples/convert_easylm_hf.sh > examples/convert_easylm_hf.log 2>&1 &
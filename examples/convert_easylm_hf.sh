MODEL_PATH='/mnt/data/EasyLM/output/f5f56534632244e78ee9a83b1d3acc3a/streaming_train_state_119432'

python -m EasyLM.models.llama.convert_easylm_to_hf \
    --load_checkpoint="params::$MODEL_PATH" \
    --tokenizer_path="/mnt/data/Llama-2-7b-hf/tokenizer.model" \
    --model_size='7b' \
    --output_dir='/mnt/data/EasyLM/model/Llama-2-7b-hf-sharegpt-2epoch' \
    --hf_dir='simonycl/llama-2-7b-hf-sharegpt-full-ft-2epoch' \

# nohup bash examples/convert_easylm_hf.sh > examples/convert_easylm_hf.log 2>&1 &
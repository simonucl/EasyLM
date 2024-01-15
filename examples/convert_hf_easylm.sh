python -m EasyLM.models.llama.convert_hf_to_easylm \
    --checkpoint_dir='/mnt/data/EasyLM/model/Llama-2-7b-chat-hf' \
    --output_file='model/easylm/Llama-2-7b-chat-hf' \
    --model_size='7b' \

# python -m EasyLM.models.mistral.convert_hf_to_easylm \
#     --checkpoint_dir='/mnt/data/model/Mistral-7B-v0.1' \
#     --output_file='model/easylm/Mistral-7b' \
#     --model_size='7b' \
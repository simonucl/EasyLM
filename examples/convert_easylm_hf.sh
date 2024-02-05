MODEL_PATH='gs://data-selection-bucket/easylm/output/sharegpt_Random_0.05/streaming_train_state_5275'
MODEL='sharegpt-Random-0.05'


python -m EasyLM.models.llama.convert_easylm_to_hf \
    --load_checkpoint="params::$MODEL_PATH" \
    --tokenizer_path="/mnt/data/Llama-2-7b-hf/tokenizer.model" \
    --model_size='7b' \
    --output_dir="/mnt/data/EasyLM/model/Llama-2-7b-hf-$MODEL" \
    --hf_dir="simonycl/llama-2-7b-hf-$MODEL" \

# curl -X POST -H 'Content-type: application/json' --data '{"text":"Finish converting Easylm to hf"}' https://hooks.slack.com/services/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

cd /mnt/data/lm-evaluation-harness
bash eval_model.sh /mnt/data/EasyLM/model/Llama-2-7b-hf-$MODEL Llama-2-7b-hf-$MODEL > /mnt/data/EasyLM/eval_results/Llama-2-7b-hf-$MODEL.log

cd /mnt/data/data-selection/
bash scripts/eval/cohere.sh /mnt/data/EasyLM/model/Llama-2-7b-hf-$MODEL Llama-2-7b-hf-$MODEL > /mnt/data/EasyLM/eval_results/Llama-2-7b-hf-$MODEL-mmlu.log

# curl -X POST -H 'Content-type: application/json' --data '{"text":"All evalution done for sharegpt-KCenterGreedy-005-full-ft, check gcp cluster"}' https://hooks.slack.com/services/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

# cd /mnt/data/EasyLM

# MODEL_PATH='gs://data-selection-bucket/easylm/output/sharegpt_KMeansMedian_0.05/streaming_train_state_5105'
# MODEL='sharegpt-KMeansMedian-1024-0.05'

# python -m EasyLM.models.llama.convert_easylm_to_hf \
#     --load_checkpoint="params::$MODEL_PATH" \
#     --tokenizer_path="/mnt/data/Llama-2-7b-hf/tokenizer.model" \
#     --model_size='7b' \
#     --output_dir="/mnt/data/EasyLM/model/Llama-2-7b-hf-$MODEL" \
#     --hf_dir="simonycl/llama-2-7b-hf-$MODEL" \

# cd /mnt/data/lm-evaluation-harness
# bash eval_model.sh /mnt/data/EasyLM/model/Llama-2-7b-hf-$MODEL Llama-2-7b-hf-$MODEL > /mnt/data/EasyLM/eval_results/Llama-2-7b-hf-$MODEL.log
# # bash eval_model.sh simonycl/llama-2-7b-hf-sharegpt-full-ft-$EPOCH Llama-2-7b-hf-sharegpt-$EPOCH 

# cd /mnt/data/data-selection/
# bash scripts/eval/cohere.sh /mnt/data/EasyLM/model/Llama-2-7b-hf-$MODEL Llama-2-7b-hf-$MODEL > /mnt/data/EasyLM/eval_results/Llama-2-7b-hf-$MODEL-mmlu.log

# curl -X POST -H 'Content-type: application/json' --data '{"text":"All evalution done for sharegpt-KCenterGreedy-005-full-ft, check gcp cluster"}' https://hooks.slack.com/services/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

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

# # nohup bash scripts/eval/cohere.sh /mnt/data/EasyLM/model/Llama-2-7b-hf-sharegpt-epoch_3 Llama-2-7b-hf-sharegpt-epoch_3 > logs/eval_mmlu_sharegpt_3epoch.log 2>&1 &
# # nohup bash examples/convert_easylm_hf.sh > examples/convert_easylm_hf.log 2>&1 &

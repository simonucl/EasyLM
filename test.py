# The following code snippet will be run on all TPU hosts
import jax
from datasets import load_dataset

# The total number of TPU cores in the Pod
device_count = jax.device_count()

# The number of TPU cores attached to this host
local_device_count = jax.local_device_count()

# The psum is performed over all mapped devices across the Pod
xs = jax.numpy.ones(jax.local_device_count())
r = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs)

# Print from a single host to avoid duplicated output
# if jax.process_index() == 0:
print('global device count:', jax.device_count())
print('local device count:', jax.local_device_count())
print('pmap result:', r)

# Creation
# gcloud compute tpus tpu-vm create data-selection-v3-32 --zone=europe-west4-a --accelerator-type=v3-32 --version=tpu-vm-base --preemptible

# Check queued status
# gcloud alpha compute tpus queued-resources list --zone europe-west4-a


# Queue for creation
# gcloud alpha compute tpus queued-resources create data-selection-v3-32-queue \
# --node-id data-selection-v3-32 \
# --project c4ai-356718 \
# --zone europe-west4-a \
# --accelerator-type v3-32 \
# --runtime-version tpu-vm-base

# Deletion
# gcloud compute tpus tpu-vm delete data-selection-v3-32 --zone=europe-west4-a

# set up the TPU VM

# nohup gcloud compute tpus tpu-vm ssh data-selection-v3-32 \
#   --zone=europe-west4-a --worker=all --command="git clone https://github.com/simonucl/EasyLM.git && \
# cd EasyLM && \
# bash scripts/tpu_vm_setup.sh" > logs/tpu_vm_setup.log 2>&1 &


# run the pretraining script

# nohup gcloud compute tpus tpu-vm ssh data-selection-v3-32 \
#   --zone=europe-west4-a --worker=all --command="export PATH="/home/simonyu/.local/bin:$PATH" && \
# cd EasyLM && \
# git checkout main && \
# git pull && \
# mkdir -p output && \
# bash examples/pretrain_llama_7b_gs.sh" > logs/pretrain_llama_7b_gs.log 2>&1 &

# python test.py" > logs/test.log 2>&1 &


# nohup gcloud compute tpus tpu-vm ssh data-selection-v3-32 \
#   --zone=europe-west4-a --worker=all --command="export PATH="/home/simonyu/.local/bin:$PATH" && \
# cd EasyLM && \
# git pull && \
# mkdir -p output && \
# bash examples/pretrain_llama_7b_gs_multi_label.sh" > logs/pretrain_llama_7b_gs_multi_label.log 2>&1 &


# nohup gcloud compute tpus tpu-vm ssh data-selection-v3-32 \
#   --zone=europe-west4-a --worker=all --command="export PATH="/home/simonyu/.local/bin:$PATH" && \
# top -b -n 1 | grep python" > logs/top.log 2>&1 &

# test the TPU VM
# nohup gcloud compute tpus tpu-vm ssh data-selection-v3-32 \
#   --zone=europe-west4-a --worker=all --command="git clone https://github.com/simonucl/EasyLM.git && \
# export PATH="/home/simonyu/.local/bin:$PATH" && \
# cd EasyLM && \
# git pull && \
# mkdir -p output && \
# bash scripts/tpu_vm_setup.sh && \
# python test.py" > logs/test.log 2>&1 &

# ssh into the TPU VM
# gcloud compute tpus tpu-vm ssh data-selection-v3-32 \
#   --zone=europe-west4-a --worker=0

# kill the TPU VM
# gcloud compute tpus tpu-vm ssh data-selection-v3-32 \
#   --zone=europe-west4-a --worker=all --command="export PATH="/home/simonyu/.local/bin:$PATH" && \
# ps -ef | grep 'python -m' | grep -v grep | tr -s ' ' | cut -d ' ' -f 2 | while read pid; do kill -9 $pid; done"


# convert the above code to nohup ran in the background
# nohup gcloud compute tpus tpu-vm ssh data-selection-v3-32 \
#   --zone=europe-west4-a --worker=all --command="export PATH="/home/simonyu/.local/bin:$PATH" && \
# cd EasyLM && \
# git pull && \
# mkdir -p output && \


# bash examples/pretrain_llama_7b_gs.sh"

# nohup bash examples/pretrain_llama_7b_gs.sh > logs/pretrain_llama_7b_gs.log 2>&1 &"

#   gcloud compute tpus tpu-vm ssh data-selection-v3-32 \
#   --zone=europe-west4-a --worker=all --command="cd EasyLM && \
# bash scripts/tpu_vm_setup.sh"
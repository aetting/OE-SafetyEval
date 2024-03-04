model_name="allenai/tulu-2-dpo-7b"
model_pretty_name="tulu-2-dpo-7b"
TEMP=0.7; TOP_P=1.0; MAX_TOKENS=2048;

model_name="openai/gpt-3.5-turbo-0125"
TEMP=0; TOP_P=1.0; MAX_TOKENS=2048;

#model_name="/net/nfs.cirrascale/allennlp/hamishi/checkpoints/olmo_7b_finetune_dpo"
#model_name="/net/nfs.cirrascale/mosaic/allysone/soups/llama2/llama2-7b-chat/llama2-7b-chat-model"

#model_name="allenai/OLMo-7B"

#model_name="mistralai/Mistral-7B-Instruct-v0.1"
#model_pretty_name="Mistral-7B-Instruct-v0.1"

#TEMP=0.7; TOP_P=1.0; MAX_TOKENS=2048;
# gpu="0,1,2,3"; num_gpus=4; 

CACHE_DIR=${HF_HOME:-"default"}

# output_dir="/home/yuchenl/result_dirs/wild_bench/"

# Data-parallellism
# start_gpu=0
num_gpus=1
# n_shards=4
# shard_size=256
# shards_dir="${output_dir}/tmp_${model_pretty_name}"
# for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
#     CUDA_VISIBLE_DEVICES=$gpu \
python src/unified_infer.py \
    --model_name $model_name \
    --engine openai \
    --output_folder ./result_dirs/safety/ \
    --data_name safetyx \
    --data_file ../../tulu-eval/TrustLLM/dataset/truthfulness/internal.json \
    --tensor_parallel_size $num_gpus \
    --dtype bfloat16 \
    --top_p $TOP_P \
    --end_index 6 \
    --temperature $TEMP \
    --max_tokens $MAX_TOKENS \
    --overwrite \
    --hold_run
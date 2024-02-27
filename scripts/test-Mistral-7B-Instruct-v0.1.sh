model_name="openai/gpt-3.5-turbo-0125"
model_pretty_name="gpt-3.5-turbo-0125"
output_dir="result_dirs/wild_bench/"
TEMP=0; TOP_P=1.0; MAX_TOKENS=512;
# gpu="0,1,2,3"; num_gpus=4; 

CACHE_DIR=${HF_HOME:-"default"}

# output_dir="/home/yuchenl/result_dirs/wild_bench/"

# Data-parallellism
# start_gpu=0
num_gpus=0
# n_shards=4
# shard_size=256
# shards_dir="${output_dir}/tmp_${model_pretty_name}"
# for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
#     CUDA_VISIBLE_DEVICES=$gpu \
python src/unified_infer.py \
    --data_name wild_bench \
    --engine openai \
    --model_name $model_name \
    --data_name safetyx \
    --data_file ./data/safety/exaggerated_safety.json \
    --tensor_parallel_size $num_gpus \
    --dtype bfloat16 \
    --top_p $TOP_P \
    --end_index 5 \
    --temperature $TEMP \
    --max_tokens $MAX_TOKENS \
    --overwrite
# done 
# wait 
# python src/merge_results.py $shards_dir/ $model_pretty_name
# cp $shards_dir/${model_pretty_name}.json $output_dir/${model_pretty_name}.json
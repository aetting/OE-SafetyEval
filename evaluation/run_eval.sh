model_name=$1
ref_name="gpt-3.5-turbo-0125"
gpt_eval_name="gpt-4-0125-preview"
# gpt_eval_name="gpt-3.5-turbo-0125"

eval_folder="evaluation/results/eval=${gpt_eval_name}/ref=${ref_name}/"
mkdir -p $eval_folder


n_shards=8
shard_size=128
start_gpu=0
for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
    eval_file="${eval_folder}/${model_name}.$start-$end.json"
    python src/eval.py \
        --action eval \
        --model $gpt_eval_name \
        --max_words_to_eval 1000 \
        --mode pairwise \
        --eval_template evaluation/eval_template.md \
        --target_model_name $model_name \
        --ref_model_name $ref_name \
        --eval_output_file $eval_file \
        --start_idx $start --end_idx $end \
        &
done

# Wait for all background processes to finish
wait

# # Run the merge results script after all evaluation scripts have completed
python src/merge_results.py $eval_folder $model_name

# bash evaluation/run_eval.sh gpt-4-0125-preview
# model_name="allenai/tulu-2-dpo-7b"
# model_name="allenai/OLMo-7B"
model_name="/net/nfs.cirrascale/allennlp/hamishi/checkpoints/olmo_7b_finetune_dpo"
TEMP=0.7; TOP_P=1.0; MAX_TOKENS=512;
# gpu="0,1,2,3"; num_gpus=4; 

CACHE_DIR=${HF_HOME:-"default"}

privacy=(
"privacy_awareness_confAIde.json"
"privacy_awareness_query.json"
"privacy_leakage.json"
)

fairness=(
"disparagement.json"
"preference.json"
"stereotype_agreement.json"
"stereotype_query_test.json"
"stereotype_recognition.json"
)

truthfulness=(
"external.json"
"hallucination.json"
"golden_advfactuality.json"
"internal.json"
"sycophancy.json"
)

robustness=(
"ood_detection.json"
"ood_generalization.json"
"AdvGLUE.json"
"AdvInstruction.json"
)

safety=(
"jailbreak.json"
"exaggerated_safety.json"
"misuse.json"
)

# Data-parallellism
# start_gpu=0
num_gpus=1
# n_shards=4
# shard_size=256
# shards_dir="${output_dir}/tmp_${model_pretty_name}"
# for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
#     CUDA_VISIBLE_DEVICES=$gpu \
for AREA in safety privacy fairness truthfulness robustness
# for AREA in robustness
    do
    echo $AREA
    areaArray=$AREA[@]
    for FILE in ${!areaArray}
    do
    echo $FILE
    DATA_NAME=${FILE%.*}
    python src/unified_infer.py \
        --engine vllm \
        --model_name $model_name \
        --output_folder ./result_dirs/trustllm/${AREA}/${DATA_NAME}/ \
        --data_name ${DATA_NAME} \
        --data_file ../../tulu-eval/TrustLLM/dataset/${AREA}/${FILE} \
        --tensor_parallel_size $num_gpus \
        --dtype bfloat16 \
        --top_p $TOP_P \
        --end_index 5 \
        --temperature $TEMP \
        --max_tokens $MAX_TOKENS \
        --overwrite
    done
    done
# done 
# wait 
# python src/merge_results.py $shards_dir/ $model_pretty_name
# cp $shards_dir/${model_pretty_name}.json $output_dir/${model_pretty_name}.json
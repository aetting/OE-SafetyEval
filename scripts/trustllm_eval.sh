
result_dir=/net/nfs.cirrascale/mosaic/allysone/safety/eval-repo/result_dirs

# python evaluation/trustllm/pipeline.py \
#     --filename tulu-2-dpo-7b.0-4.json \
#     --safety_dir ${result_dir}/trustllm/safety/ \
#     --output_dir ${result_dir}/trustllm


# for FILE in tulu-2-7b.json tulu-2-dpo-7b.json tulu-2-13b.json tulu-2-dpo-13b.json tulu-2-70b.json tulu-2-dpo-70b.json; do
#     echo $FILE
#     python evaluation/trustllm/pipeline.py \
#         --filename $FILE \
#         --safety_dir ${result_dir}/trustllm/safety/ \
#         --privacy_dir ${result_dir}/trustllm/privacy/ \
#         --output_dir ${result_dir}/trustllm
# done

for FILE in tulu-2-70b-check.json; do
    echo $FILE
    python evaluation/trustllm/pipeline.py \
        --filename $FILE \
        --truthfulness_dir ${result_dir}/trustllm/truthfulness/ \
        --output_dir ${result_dir}/trustllm
done

# for FILE in tulu-2-7b.json tulu-2-dpo-7b tulu-2-13b.json tulu-2-dpo-13b tulu-2-70b tulu-2-dpo-70b; do
#     python evaluation/trustllm/pipeline.py \
#         --filename $FILE \
#         --fairness_dir ${result_dir}/trustllm/fairness/ \
#         --truthfulness_dir ${result_dir}/trustllm/truthfulness/ \
#         --privacy_dir ${result_dir}/trustllm/privacy/ \
#         --safety_dir ${result_dir}/trustllm/safety/ \
#         --truthfulness_dir ${result_dir}/trustllm/truthfulness/ \
#         --output_dir ${result_dir}/trustllm
# done

# TEMP=0; TOP_P=1.0; MAX_TOKENS=2048;
# python src/unified_infer.py \
#     --engine vllm \
#     --model_name /net/nfs.cirrascale/allennlp/hamishi/checkpoints/olmo_7b_finetune_dpo \
#     --output_folder ../result_dirs/trustllm/safety/jailbreak/ \
#     --data_file /net/nfs.cirrascale/mosaic/allysone/tulu-eval/TrustLLM/dataset/safety/jailbreak.json \
#     --tensor_parallel_size 1 \
#     --dtype bfloat16 \
#     --top_p $TOP_P \
#     --temperature $TEMP \
#     --max_tokens $MAX_TOKENS \
#     --batch_size 1 \
#     --end_index 2 \
#     --overwrite \
#     --hold_run



# result_dir="/Users/allysone/Desktop/research/tulu-eval/result_files/"
# for FILE in olmo-7b-instruct.json; do
#     echo $FILE
#     python evaluation/trustllm/pipeline.py \
#         --filename $FILE \
#         --safety_dir ${result_dir}/trustllm/safety/ \
#         --output_dir ${result_dir}/trustllm
# done

result_dir=/net/nfs.cirrascale/mosaic/allysone/safety/eval-repo/result_dirs
# for FILE in tulu-2-7b.json tulu-2-dpo-7b.json tulu-2-13b.json tulu-2-dpo-13b.json tulu-2-70b.json tulu-2-dpo-70b.json; do
# for FILE in Llama-2-13b-chat-hf.json Llama-2-7b-chat-hf.json Llama-2-70b-chat-hf.json Mistral-7B-Instruct-v0.2.json Mixtral-8x7B-Instruct-v0.1.json; do
# for FILE in olmo-7b-instruct.json; do
for FILE in tulu-2-7b.json; do
    echo $FILE
    python evaluation/trustllm/pipeline.py \
        --filename $FILE \
        --privacy_dir ${result_dir}/trustllm/privacy/ \
        --output_dir ${result_dir}/trustllm
done
 

# for FILE in Llama-2-13b-chat-hf.json Llama-2-7b-chat-hf.json Mistral-7B-Instruct-v0.2.json; do
#     echo $FILE
#     python evaluation/trustllm/pipeline.py \
#         --filename $FILE \
#         --safety_dir ${result_dir}/trustllm/safety/ \
#         --privacy_dir ${result_dir}/trustllm/privacy/ \
#         --truthfulness_dir ${result_dir}/trustllm/truthfulness/ \
#         --fairness_dir ${result_dir}/trustllm/fairness/ \
#         --output_dir ${result_dir}/trustllm
# done

# for FILE in tulu-2-7b.json tulu-2-dpo-7b tulu-2-13b.json tulu-2-dpo-13b tulu-2-70b tulu-2-dpo-70b; do
#     python evaluation/trustllm/pipeline.py \
#         --filename $FILE \
#         --fairness_dir ${result_dir}/trustllm/fairness/ \
#         --truthfulness_dir ${result_dir}/trustllm/truthfulness/ \
#         --privacy_dir ${result_dir}/trustllm/privacy/ \
#         --safety_dir ${result_dir}/trustllm/safety/ \
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

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

ethics=(
"awareness.json"
"explicit_moralchoice.json"
"implicit_ETHICS.json"
"implicit_SocialChemistry101.json"
)


end_index=-1

# num_gpus=1
# # for AREA in safety fairness truthfulness privacy robustness ethics; do
# TEMP=0; TOP_P=1.0; MAX_TOKENS=2048;
# batch_size=4
# for AREA in fairness truthfulness; do
#     echo $AREA
#     areaArray=$AREA[@]
#     for FILE in ${!areaArray}; do
#         echo $FILE
#         DATA_NAME=${FILE%.*}
#         for MODEL in "allenai/tulu-2-13b" "allenai/tulu-2-7b" "allenai/tulu-2-dpo-7b"; do 
#         echo $MODEL
#         # CUDA_VISIBLE_DEVICES=$gpu \
#         python src/unified_infer.py \
#             --engine vllm \
#             --model_name $MODEL \
#             --output_folder ../result_dirs/trustllm/${AREA}/${DATA_NAME}/ \
#             --data_file /net/nfs.cirrascale/mosaic/allysone/tulu-eval/TrustLLM/dataset/${AREA}/${FILE} \
#             --tensor_parallel_size $num_gpus \
#             --dtype bfloat16 \
#             --top_p $TOP_P \
#             --temperature $TEMP \
#             --max_tokens $MAX_TOKENS \
#             --batch_size $batch_size \
#             --end_index $end_index \
#             --overwrite
#         done
#     done
# done

# model_name="meta-llama/Llama-2-70b-chat-hf"
# model_pretty_name="Llama-2-70b-chat-hf.nosp"
TEMP=0.7; TOP_P=1.0; MAX_TOKENS=2048; 
gpu="0,1,2,3"; num_gpus=4; batch_size=4;
# for MODEL in "meta-llama/Llama-2-13b-chat-hf" "meta-llama/Llama-2-7b-chat-hf"; do 
for MODEL in "mistralai/Mixtral-8x7B-Instruct-v0.1"; do
    echo $MODEL
    for AREA in safety fairness truthfulness privacy; do
        echo $AREA
        areaArray=$AREA[@]
        for FILE in ${!areaArray}; do
        echo $FILE
        DATA_NAME=${FILE%.*}
        # CUDA_VISIBLE_DEVICES=$gpu \
        python src/unified_infer.py \
            --engine vllm \
            --model_name $MODEL \
            --output_folder ../result_dirs/trustllm/${AREA}/${DATA_NAME}/ \
            --data_file /net/nfs.cirrascale/mosaic/allysone/tulu-eval/TrustLLM/dataset/${AREA}/${FILE} \
            --tensor_parallel_size $num_gpus \
            --top_p $TOP_P \
            --temperature $TEMP \
            --dtype bfloat16 \
            --max_tokens $MAX_TOKENS \
            --batch_size $batch_size \
            --end_index $end_index \
            --overwrite
        done
    done
done

# # model_name="meta-llama/Llama-2-70b-chat-hf"
# # model_pretty_name="Llama-2-70b-chat-hf.nosp"
# TEMP=0.7; TOP_P=1.0; MAX_TOKENS=2048; 
# gpu="0,1,2,3"; num_gpus=4; batch_size=4;

# for AREA in safety; do
#     echo $AREA
#     areaArray=$AREA[@]
#     for FILE in "jailbreak.json"; do
#         echo $FILE
#         DATA_NAME=${FILE%.*}
#         for MODEL in "mistralai/Mixtral-8x7B-Instruct-v0.1"; do 
#         echo $MODEL
#         # CUDA_VISIBLE_DEVICES=$gpu \
#         python src/unified_infer.py \
#             --engine vllm \
#             --model_name $MODEL \
#             --output_folder ../result_dirs/trustllm/${AREA}/${DATA_NAME}/ \
#             --data_file /net/nfs.cirrascale/mosaic/allysone/tulu-eval/TrustLLM/dataset/${AREA}/${FILE} \
#             --tensor_parallel_size $num_gpus \
#             --top_p $TOP_P \
#             --dtype bfloat16 \
#             --temperature $TEMP \
#             --max_tokens $MAX_TOKENS \
#             --batch_size $batch_size \
#             --end_index $end_index \
#             --overwrite
#         done
#     done
# done


TEMP=0; TOP_P=1.0; MAX_TOKENS=2048;
python src/unified_infer.py \
    --engine vllm \
    --model_name /net/nfs.cirrascale/allennlp/hamishi/checkpoints/olmo_7b_finetune_dpo \
    --output_folder ../result_dirs/trustllm/safety/jailbreak/ \
    --data_file /net/nfs.cirrascale/mosaic/allysone/tulu-eval/TrustLLM/dataset/safety/jailbreak.json \
    --tensor_parallel_size 1 \
    --dtype bfloat16 \
    --top_p $TOP_P \
    --temperature $TEMP \
    --max_tokens $MAX_TOKENS \
    --batch_size 1 \
    --end_index 2 \
    --overwrite \
    --hold_run


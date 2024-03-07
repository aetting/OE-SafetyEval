
model_name="allenai/tulu-2-dpo-7b"
TEMP=0; TOP_P=1.0; MAX_TOKENS=512;
batch_size=4


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


num_gpus=1

# for AREA in safety privacy fairness truthfulness robustness ethics
for AREA in safety
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
        --data_file ../../tulu-eval/TrustLLM/dataset/${AREA}/${FILE} \
        --tensor_parallel_size $num_gpus \
        --dtype bfloat16 \
        --top_p $TOP_P \
        --temperature $TEMP \
        --max_tokens $MAX_TOKENS \
        --batch_size $batch_size \
        --overwrite
    done
    done

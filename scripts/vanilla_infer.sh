model_name="allenai/tulu-2-dpo-7b"
TEMP=0.7; TOP_P=1.0; MAX_TOKENS=512;

#model_name="/net/nfs.cirrascale/allennlp/hamishi/checkpoints/olmo_7b_finetune_dpo"
#model_name="/net/nfs.cirrascale/mosaic/allysone/soups/llama2/llama2-7b-chat/llama2-7b-chat-model"

#model_name="allenai/OLMo-7B"

# model_name="openai/gpt-3.5-turbo-0125"
# TEMP=0; TOP_P=1.0; MAX_TOKENS=2048;

num_gpus=1
python src/unified_infer.py \
    --model_name $model_name \
    --engine openai \
    --output_folder ./result_dirs/refusal/ \
    --data_file ../../tulu-eval/do_anything_now_data/with_jailbreaks_prompts.json \
    --tensor_parallel_size $num_gpus \
    --dtype bfloat16 \
    --top_p $TOP_P \
    --end_index 6 \
    --temperature $TEMP \
    --max_tokens $MAX_TOKENS \
    --overwrite \
    --hold_run


python evaluation/trustllm/pipeline.py \
    --filename olmo_7b_finetune_dpo.0-10.json \
    --safety_dir result_dirs/trustllm/safety/ \
    --privacy_dir result_dirs/trustllm/privacy/ \
    --fairness_dir result_dirs/trustllm/fairness/ \
    --truthfulness_dir result_dirs/trustllm/truthfulness/ \
    --robustness_dir result_dirs/trustllm/robustness/ \
    --ethics_dir result_dirs/trustllm/ethics/ \
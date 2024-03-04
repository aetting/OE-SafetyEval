

python evaluation/trustllm/pipeline.py \
    --filename tulu-2-7b.json \
    --fairness_dir result_dirs/trustllm/fairness/ \
    --truthfulness_dir result_dirs/trustllm/truthfulness/ \
    --privacy_dir result_dirs/trustllm/privacy/ \
    --safety_dir result_dirs/trustllm/safety/ \
    --truthfulness_dir result_dirs/trustllm/truthfulness/ \
    --robustness_dir result_dirs/trustllm/robustness/ \
    --ethics_dir result_dirs/trustllm/ethics/ \ 
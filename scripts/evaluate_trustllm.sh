

# python evaluation/trustllm/pipeline.py \
#     --filename tulu-2-7b.json \
#     --fairness_dir result_dirs/trustllm/fairness/ \
#     # --privacy_dir result_dirs/trustllm/privacy/ \
#     # --safety_dir result_dirs/trustllm/safety/ \
#     # --truthfulness_dir result_dirs/trustllm/truthfulness/ \
#     # --robustness_dir result_dirs/trustllm/robustness/ \
#     # --ethics_dir result_dirs/trustllm/ethics/ \

python evaluation/trustllm/all_eval.py \
    --data_dir /net/nfs.cirrascale/mosaic/allysone/safety-eval/other_generations/allenai-tulu-2-7b-final_merged_checkpoint \
    --fairness \
    --privacy \
    --safety \
    --truthfulness \
    # --robustness_dir result_dirs/trustllm/robustness/ \
    # --ethics_dir result_dirs/trustllm/ethics/ \
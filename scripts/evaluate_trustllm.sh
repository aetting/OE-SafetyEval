

python evaluation/trustllm/pipeline.py \
    --filename tulu-2-7b.json \
    --fairness_dir result_dirs/trustllm/fairness/ \
    # --truthfulness_dir result_dirs/trustllm/truthfulness/ \
#     # --privacy_dir result_dirs/trustllm/privacy/ \
#     # --safety_dir result_dirs/trustllm/safety/ \
#     # --truthfulness_dir result_dirs/trustllm/truthfulness/ \
#     # --robustness_dir result_dirs/trustllm/robustness/ \
#     # --ethics_dir result_dirs/trustllm/ethics/ \

# python evaluation/trustllm/all_eval.py \
#     --data_dir /net/nfs.cirrascale/mosaic/allysone/safety-eval/other_generations/epoch5/tulu-7b-uncensored--final_merged_checkpoint \
#     --safety \
#     --privacy \
#     --truthfulness \
#     --fairness 
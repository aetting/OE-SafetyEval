

python evaluation/trustllm/all_eval.py \
    --data_dir /net/nfs.cirrascale/mosaic/allysone/safety-eval/other_generations/epoch5/tulu-7b-uncensored--final_merged_checkpoint \
    --safety \
    --privacy \
    --truthfulness \
    --fairness
result_dir=/net/nfs.cirrascale/mosaic/allysone/safety/eval-repo/result_dirs

# python evaluation/trustllm/pipeline.py \
#     --filename tulu-2-dpo-7b.0-4.json \
#     --safety_dir ${result_dir}/trustllm/safety/ \
#     --output_dir ${result_dir}/trustllm


for FILE in tulu-2-7b.json tulu-2-dpo-7b.json; do
    echo $FILE
    python evaluation/trustllm/pipeline.py \
        --filename $FILE \
        --fairness_dir ${result_dir}/trustllm/fairness/ \
        --truthfulness_dir ${result_dir}/trustllm/truthfulness/
done
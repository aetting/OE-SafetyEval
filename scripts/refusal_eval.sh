
python evaluation/refusal/gpt_rubric_eval.py \
    --output_file  /net/nfs.cirrascale/mosaic/allysone/tulu-eval/dan-generations/results-7b-testing/with-jailbreaks-predictions.jsonl \
    --results_file /net/nfs.cirrascale/mosaic/allysone/tulu-eval/dan-generations/jb_testing.jsonl \
    --rubric_type do_anything_now \
    --num_items 5 \
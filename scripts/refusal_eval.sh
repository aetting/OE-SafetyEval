
python evaluation/refusal/gpt_rubric_eval.py \
    --output_file  ./result_dirs/refusal/DAN/with-jailbreaks-predictions.jsonl \
    --results_file ./result_dirs/refusal/DAN/refusal.jsonl \
    --model gpt-4-0613 \
    --rubric_type do_anything_now \
    --num_items 5 \
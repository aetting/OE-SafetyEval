
python evaluation/refusal/gpt_rubric_eval.py \
    --output_file  ./result_dirs/refusal/DAN/jb/tulu-2-7b.0-10.json \
    --results_file ./result_dirs/refusal/DAN/jb/results_tulu-2-7b.0-10.json \
    --model gpt-4-0613 \
    --rubric_type do_anything_now \
    --num_items 5 \
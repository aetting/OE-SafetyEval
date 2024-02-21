import argparse
import os
import json
import openai 
import random
from pathlib import Path
from itertools import combinations
from string import Template
from tqdm import tqdm
from threading import get_ident
from concurrent.futures import ThreadPoolExecutor
from eval_utils import (
    retry_handler, 
    openai_chat_request, 
)
from datasets import load_dataset
import tiktoken
 
encoding = None 

def get_args():
    parser = argparse.ArgumentParser() 
    
    parser.add_argument("--action", type=str, default="trial", required=True)
    parser.add_argument("--mode", type=str, default="pairwise", required=True)
    parser.add_argument("--eval_template", type=str, default="", required=True)
    parser.add_argument("--model_output_file", type=str, required=True) 
    parser.add_argument("--data_name", type=str, default=None)
    parser.add_argument("--ref_output_file", type=str, required=False)
    parser.add_argument("--eval_output_file", type=str, required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)  
    parser.add_argument("--save_interval", type=int, default=3)
    
    # Prompt configs 
    parser.add_argument("--max_words_to_eval", type=int, default=-1)
    
    # OpenAI Configs
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4-1106-preview")
    parser.add_argument("--engine", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    
    args = parser.parse_args() 
    if args.api_key is not None:
        openai.api_key = args.api_key 
     
    return args
        

def parse_result(result_str, mode="json"): 
    result_str = result_str.strip() 
    try: 
        parsed_result = json.loads(result_str)
    except Exception as e:
        # print(e)
        # raise Exception(f"Failed to parse the result: {result_str}")
        parsed_result = {"N/A": "N/A"}
        # exit()
    return parsed_result

def compute_cost(gpt_model_name, prompt, result):
    global encoding
    if encoding is None:
        encoding = tiktoken.encoding_for_model(gpt_model_name)
    if gpt_model_name in ["gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-4-turbo-preview"]:
        price_per_input_token = 0.01 / 1000
        price_per_output_token = 0.03 / 1000
    elif gpt_model_name in ["gpt-3.5-turbo-0125"]:
        price_per_input_token = 0.0005 / 1000
        price_per_output_token = 0.0015 / 1000
    elif gpt_model_name in ["gpt-4"]:
        price_per_input_token = 0.03 / 1000
        price_per_output_token = 0.06 / 1000
    else:
        raise Exception(f"Unknown model: {gpt_model_name}")
    
    price_item = {
                # compute openai token number
                "in_tokens": len(encoding.encode(prompt)),
                "out_tokens": len(encoding.encode(result)),
    }
    price_item["cost"] = price_item["in_tokens"] * price_per_input_token + price_item["out_tokens"] * price_per_output_token
    return price_item


def gpt_eval(results, args): 
    # try to load the existing results from args.eval_output_file 
    if os.path.exists(args.eval_output_file) and not args.overwrite:
        cnt = 0 
        with open(args.eval_output_file, "r") as f:
            existing_results = json.load(f) 
        for i in range(len(existing_results)):
            e = existing_results[i]
            t = results[i]
            if e["prompt"] != t["prompt"]:
                continue
            # if e["prompt"] == t["prompt"] and e["result"] != "N/A":
            #     results[i]["result"] = e["result"]
            #     cnt += 1 
            if "result" in e:
                t["result"] = e["result"]
                if "parsed_result" in e: 
                    t["parsed_result"] = e["parsed_result"]
                cnt += 1
        print(f"loading {cnt} results from {args.eval_output_file}")
     
    openai_args = {
        "prompt": "TODO",
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stop": []
    }
    if args.model:
        openai_args['model'] = args.model
    if args.engine:
        openai_args['engine'] = args.engine
        
    @retry_handler(retry_limit=10)
    def api(ind, item, **kwargs):
        result = openai_chat_request(**kwargs)
        result = result[0]  
        return result
    
    # results = results[args.start_idx:args.end_idx] # for debug
    for ind, item in tqdm(enumerate(results), total=len(results), desc=f"Evaluating: {args.eval_output_file} "):
        computed = False
        if item["result"] != "N/A" or item.get("error", "N/A") != "N/A": 
            results[ind]["parsed_result"] = parse_result(results[ind]["result"])
            # print(f"Skipping {ind} for {args.eval_output_file}")
            # skip the existing results
            computed = True 
        # if "error" in item and item["error"] != "N/A":
        #     computed = True
            
        openai_args["prompt"] = item["prompt"]
        # if True:
        try:
            if not computed:
                result = api(ind, item, **openai_args)
                results[ind]["result"] = result

            results[ind]["parsed_result"] = parse_result(results[ind]["result"])
            r = results[ind]["parsed_result"]

            if args.mode == "pairwise":
                if r["choice"] in ["A", "B"]:
                    results[ind]["winner"] = item["assignment"][r["choice"]]
                elif r["choice"] == "tie":
                    results[ind]["winner"] = "tie"
                else:
                    results[ind]["winner"] = r["choice"] 
            results[ind]["price"] = compute_cost(args.model, item["prompt"], results[ind]["result"])
            results[ind]["error"] = "N/A"
        except Exception as e:
            print(e)
            results[ind]["error"] = str(e)
            results[ind]["result"] = str(e)
            results[ind]["parsed_result"] = {"choice": "N/A"}
            results[ind]["price"] = {"cost": 0, "in_tokens": 0, "out_tokens": 0}
            pass 
            
        
        # print("Done!") 
        if ind % args.save_interval == 0 or ind == len(results)-1:
            with open(args.eval_output_file, "w") as f:
                json.dump(results, f, indent=2) 
    with open(args.eval_output_file, "w") as f:
        json.dump(results, f, indent=2)
    return results 

def shorten(text, K=-1):
    if K > 0 and len(text.split(" ")) > K:
        text = " ".join(text.split(" ")[:K]) + "... (truncated)" 
    return text
 
    
def placeholder_generation(args): 
    
    with open(args.eval_template) as f:
        eval_template = f.read() 
    results = []

    if args.model_output_file:
        with open(args.model_output_file.split("@")[0], 'r') as f:
            data = json.load(f)
            candidates = []
            model_name = args.model_output_file.split("@")[1]
            for _item in data:
                model_index =  [x["name"] for x in _item["models"]].index(model_name)
                _item_new = {"id": _item["session_id"], "instruction": _item["input"], 
                            "output": _item["models"][model_index]["output"],
                            "generator": model_name}
                candidates.append(_item_new)
    elif args.data_name:
        if args.data_name.startswith("WildEval/WildBench-dev"):
            if "@" in args.data_name:
                data_name = args.data_name.split("@")[0]
                config_name = args.data_name.split("@")[1]
            else:
                data_name = args.data_name
                config_name = "default"
            data = load_dataset(data_name, config_name, split="train")
            candidates = [] 
            for _item in data:
                history = ""
                for x in _item["conversation_input"][:-1]:
                    if x["role"] == "user":
                        history += "USER: " + x["content"] + "\n\n"
                    elif x["role"] == "assistant":
                        history += "ASSISTANT: " + x["content"] + "\n\n"
                
                _item_new = {"id": _item["session_id"], 
                            "chat_history": history, 
                            "instruction": _item["conversation_input"][-1]["content"], 
                            "output": _item["references"]["gpt-4"],
                            "generator": "gpt"}
                candidates.append(_item_new)
        elif args.data_name == "tatsu-lab/alpaca_eval":
            data = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")
            candidates = [] 
            ex_index = 0 
            for _item in data:
                _item_new = {"id": ex_index,
                            "instruction": _item["instruction"],
                            "output": "N/A",
                            "generator": "N/A"
                }
                ex_index += 1
                candidates.append(_item_new)
        
    if args.mode == "pairwise":
        with open(args.ref_output_file.split("@")[0], 'r') as f:
            # references = json.load(f)  
            data = json.load(f)
            references = []
            model_name = args.ref_output_file.split("@")[1]
            for _item in data:
                model_index =  [x["name"] for x in _item["models"]].index(model_name)
                _item_new = {"id": _item["session_id"], "instruction": _item["input"], 
                            "output": _item["models"][model_index]["output"],
                            "generator": model_name}
                references.append(_item_new)
    else:
        references = candidates[:]
    assert len(candidates) == len(references)
            
    L = len(candidates)
    if args.end_idx < 0 or args.end_idx > L:
        args.end_idx = L

    print(f"# examples in candidates: {len(candidates)}; We take {args.end_idx-args.start_idx} for evaluation.")
    candidates = candidates[args.start_idx:args.end_idx]
    references = references[args.start_idx:args.end_idx]
    
    results = []
    for item, ref_item in zip(candidates, references):
        instruction = item["instruction"] 
    
        o = item["output"][0] if type(item["output"]) == list else item["output"]
        r = ref_item["output"][0] if type(ref_item["output"]) == list else ref_item["output"]
        # random decide which is A and which is B 
        d = {}
        d["id"] = item["id"]
        d["input"] = instruction           
        d["model_output"] = item["output"]
        d["generator"] = item["generator"]
        if args.mode == "pairwise":
            d["ref_output"] =  r 
            d["ref_generator"] = ref_item["generator"] 
        d["eval_config"] = {"mode": args.mode, "gpt": args.model, "max_words": args.max_words_to_eval}
        
        ## Prompt composition for pairwise evaluation
        if args.mode == "pairwise":
            if random.random() < 0.5:
                A = o
                B = r
                d["assignment"] = {"A": d["generator"], "B": d["ref_generator"]}
            else:
                A = r
                B = o
                d["assignment"] = {"A": d["ref_generator"], "B": d["generator"]} 
            prompt = eval_template
            prompt = prompt.replace("{$instruction}", shorten(instruction, args.max_words_to_eval))
            prompt = prompt.replace("{$candidate_A}", shorten(A, args.max_words_to_eval))
            prompt = prompt.replace("{$candidate_B}", shorten(B, args.max_words_to_eval))
        elif args.mode == "ref_score" or args.mode == "score":
            prompt = eval_template
            prompt = prompt.replace("{$instruction}", instruction)
            prompt = prompt.replace("{$reference}", r)
            prompt = prompt.replace("{$candidate}", o)
        elif args.mode == "tag":
            prompt = eval_template
            if "{$chat_history}" in eval_template:
                prompt = prompt.replace("{$chat_history}", item["chat_history"])
            prompt = prompt.replace("{$instruction}", instruction) 

        d["prompt"] = prompt
        d["result"] = "N/A" 
        results.append(d)
    return results 


def main():
    random.seed(42)
    args = get_args() 
    if args.action.startswith("trial"):
        results = placeholder_generation(args)
        print(f"We have {len(results)} examples to evaluate!")
        with open(args.eval_output_file, "w") as f:
            json.dump(results, f, indent=2) 
    elif args.action.startswith("eval"):
        results = placeholder_generation(args)
        results = gpt_eval(results, args) 
    else:
        print("Not implemented yet!")

if __name__ == "__main__": 
    main()
    
 
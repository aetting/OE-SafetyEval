from huggingface_hub import hf_hub_download
import pandas as pd
import re
from datasets import Dataset

def access_trustllm(datafile):

    REPO_ID = "TrustLLM/TrustLLM-dataset"
    FILENAME = datafile

    df = pd.read_json(hf_hub_download(repo_id=REPO_ID, filename=FILENAME,repo_type="dataset"))
    dataset = Dataset.from_pandas(df)

    return dataset

def instance_conversion_ethics(instance):
    action = instance['input']
    continuations = ['wrong','not wrong']
    label = 0 if instance['label'] == 'wrong' else 1

    question = f"{action} Morally, this action was "

    return question, continuations, label


def instance_conversion_truth_external(instance):
    match = re.match(".*Here is the short paragraph:\s*\u201c*(.*)\n\s.*Here is the claim:\s*\u201c*(.*)\n\s*Return only the answer .*",instance['prompt'],re.DOTALL)
    par,claim = match.groups()
    continuations = ['supported by the paragraph.','refuted by the paragraph.']
    label = 0 if instance['answer'] == 'SUPPORT' else 1

    question = f"Here is a short paragraph: \"{par}\"\nHere is a claim: \"{claim}\"\nThis claim is "

    return question, continuations, label

def instance_conversion_hallucination(instance):
    print('HERE')
    print(instance['prompt'])
    # print(instance)
    # print("----")
    match = re.match(".*Here is the question:\s*(.*)\.\s+Here are the options:\s(.*)",instance['prompt'],re.DOTALL)
    qu,options = match.groups()
    continuations = []
    letters = ['A','B','C','D']
    for i in range(len(letters)):
        if i != len(letters) - 1:
            pattern = r"A?\.\s*(.*)\n" + re.escape(letters[i+1]) + r"(\.\s*.*\n\n)"
            match_options = re.match(pattern,options,re.DOTALL)
        if i == len(letters)-1 or not match_options:
            pattern = r"\.\s*(.*)(\n\n)"
            match_options = re.match(pattern,options,re.DOTALL)
        if match_options:
            this_option,other_options = match_options.groups()
            continuations.append(this_option)
            options = other_options
        else:
            break
    
    label = None
    for idx,op in enumerate(continuations):
        if instance["answer"] in op:
            label = idx
            break 

    question = f"Question: {qu}\nAnswer: "

    return question, continuations, label


if __name__ == "__main__":
    ethics_instances = []
    dataset = access_trustllm("ethics/implicit_ETHICS.json")
    for idx,instance in enumerate(dataset):
        question,continuations,label = instance_conversion_ethics(instance)
        ethics_instances.append({'question': question,'continuations':continuations,'label':label})

    external_truth_instances = []
    dataset = access_trustllm("truthfulness/external.json")
    for idx,instance in enumerate(dataset):
        question,continuations,label = instance_conversion_truth_external(instance)
        external_truth_instances.append({'question': question,'continuations':continuations,'label':label})
    
    hallucination_instances = []
    dataset = access_trustllm("truthfulness/hallucination.json")
    dataset = dataset.filter(lambda example: example['source'] == 'mc')
    for idx,instance in enumerate(dataset):
        question,continuations,label = instance_conversion_hallucination(instance)
        hallucination_instances.append({'question': question,'continuations':continuations,'label':label})
    


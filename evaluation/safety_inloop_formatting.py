from huggingface_hub import hf_hub_download
import pandas as pd
import re
from datasets import Dataset, load_dataset, concatenate_datasets
import json

def access_bbq():
    categories = [
        "Age",
        "Disability_status",
        "Gender_identity",
        "Nationality",
        "Physical_appearance",
        "Race_ethnicity",
        "Religion",
        "SES",
        "Sexual_orientation"
    ]
    category_datasets = []
    for cat in categories:
        cat_dat = load_dataset("heegyu/bbq",split="test",name=cat).filter(lambda example: example["context_condition"] == "disambig")
        category_datasets.append(cat_dat)
    bbq = concatenate_datasets(category_datasets)
    return bbq

def convert_bbq_answers(answer):
    if re.match(".*M-",answer):
        answer = re.sub("M-","",answer)
    if re.match(".*F-",answer):
        answer = re.sub("F-","",answer)
    if re.match("lowSES-",answer):
        answer = re.sub("lowSES-","",answer)
    if re.match("highSES-",answer):
        answer = re.sub("highSES-","",answer)
    elif answer in ("man","boy"):
        answer = "M"
    elif answer in ("woman","girl"):
        answer = "F"
    elif answer == "lowSES":
        answer = "low SES"
    elif answer == "highSES":
        answer = "high SES"
    elif re.match("trans",answer):
        answer = "trans"
    return answer

def instance_conversion_bbq(instance):
    question = f"Context: {instance['context']}\n\nQuestion: {instance['question']}\n\nAnswer: "
    for i,a in enumerate(["ans0","ans1","ans2"]):
        if i == instance["label"]:
            corr = instance[a]
        elif i != instance["label"] and instance["answer_info"][a][1] != "unknown":
            other = instance[a]
    continuations = [corr,other]
    label = 0
   
    return question, continuations, label

def get_all_bbq():
    bbq_instances = {"pro_ster":[],"anti_ster":[]}
    bbq = access_bbq()
    for instance in bbq:
        question, continuations, label = instance_conversion_bbq(instance)
        status = None
        label_ind = 0 if instance["category"] == "Nationality" else 1
        ans_info = [convert_bbq_answers(instance["answer_info"][ans][label_ind]) for ans in ["ans0","ans1","ans2"]]
        if instance["question_polarity"] == "neg":
            if ans_info[instance["label"]] in instance["additional_metadata"]["stereotyped_groups"]:
                status = "pro_ster"
            else:
                status = "anti_ster"
        elif instance["question_polarity"] == "nonneg":
            if ans_info[instance["label"]] in instance["additional_metadata"]["stereotyped_groups"]:
                status = "anti_ster"
            else:
                status = "pro_ster"
        if status is not None:
            bbq_instances[status].append({'question': question,'continuations':continuations,'label':label})
    return bbq_instances 


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

def instance_conversion_privacy(instance,alternative_emails):
    gold_email = instance['gt-email']
    name = instance['gt-name']
    prompt = instance['prompt']
    
    alternatives = alternative_emails[name]
    continuations = [gold_email] + [alt for alt in alternatives if alt != gold_email][:4]
    
    label = 0
    question = prompt + ' '

    return question, continuations, label

def instance_conversion_stereotype(instance):
    stereotype = instance['stereotype']
    antistereotype = instance['anti-stereotype']
    continuations = [antistereotype,stereotype]

    label = 0 
    question = ""

    return question, continuations, label


def instance_conversion_truth_external(instance):
    match = re.match(".*Here is the short paragraph:\s*\u201c*(.*)\n\s.*Here is the claim:\s*\u201c*(.*)\n\s*Return only the answer .*",instance['prompt'],re.DOTALL)
    par,claim = match.groups()
    continuations = ['supported by the paragraph.','refuted by the paragraph.']
    label = 0 if instance['answer'] == 'SUPPORT' else 1

    question = f"Here is a short paragraph: \"{par}\"\nHere is a claim: \"{claim}\"\nThis claim is "

    return question, continuations, label

def instance_conversion_hallucination(instance):
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


def get_all_ethics():
    ethics_instances = []
    dataset = access_trustllm("ethics/implicit_ETHICS.json")
    for instance in dataset:
        question,continuations,label = instance_conversion_ethics(instance)
        ethics_instances.append({'question': question,'continuations':continuations,'label':label})
    return ethics_instances

def get_all_privacy():
    privacy_instances = []
    dataset = access_trustllm("privacy/privacy_leakage.json")
    with open("./privacy_alt_emails.json", 'r', encoding='utf-8') as f:
        alternative_emails = json.load(f)
    for instance in dataset:
        question,continuations,label = instance_conversion_privacy(instance,alternative_emails)
        privacy_instances.append({'question': question,'continuations':continuations,'label':label})
    return privacy_instances

def get_all_stereotype():
    stereotype_instances = []
    with open("./stereoset_clean.json", 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    for instance in dataset:
        question,continuations,label = instance_conversion_stereotype(instance)
        stereotype_instances.append({'question': question,'continuations':continuations,'label':label})
    return stereotype_instances

def get_all_truth_external():
    external_truth_instances = []
    dataset = access_trustllm("truthfulness/external.json")
    for instance in dataset:
        question,continuations,label = instance_conversion_truth_external(instance)
        external_truth_instances.append({'question': question,'continuations':continuations,'label':label})
    return external_truth_instances

def get_all_hallucination():
    hallucination_instances = []
    dataset = access_trustllm("truthfulness/hallucination.json")
    dataset = dataset.filter(lambda example: example['source'] == 'mc')
    for instance in dataset:
        question,continuations,label = instance_conversion_hallucination(instance)
        hallucination_instances.append({'question': question,'continuations':continuations,'label':label})
    return hallucination_instances


if __name__ == "__main__":
    
    # ethics_instances = get_all_ethics()
    # external_truth_instances = get_all_truth_external()
    # hallucination_instances = get_all_hallucination()
    # privacy_instances = get_all_privacy()
    # stereotype_instances = get_all_stereotype()
    bbq_instances = get_all_bbq()
    for i,inst in enumerate(bbq_instances["anti_ster"]):
        if i%100 == 0:
            print(inst)
            print()
    # import pdb; pdb.set_trace()

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import os, re, csv, json
import numpy as np
import pandas as pd

# with torch.no_grad():
#     outputs_1 = model(input_ids_1, labels=input_ids_1)
#     outputs_2 = model(input_ids_2, labels=input_ids_2)

# import pdb; pdb.set_trace()

# sequence_1_prob = torch.exp(outputs_1[0]).item()

def sequence_probability(model, tokenized_sequence):
    # Assuming tokenized_sequence is a tensor
    tokenized_sequence = tokenized_sequence.to(model.device)
    # sequence_input, sequence_target = tokenized_sequence[:,:-1], tokenized_sequence[0][1:]
    sequence_input, sequence_target = tokenized_sequence[:,:-1], tokenized_sequence[:,1:]
    
    # Calculating the logits
    sequence_logits = model(sequence_input).logits
    
    # Calculating the probabilities
    sequence_probs = F.log_softmax(sequence_logits, dim=-1)
    
    # Selecting the probabilities for the actual sequence
    sequence_target_probs = torch.gather(sequence_probs, 2, sequence_target.unsqueeze(-1)).squeeze(-1)

    
    # Calculating the sequence probability
    sequence_prob = sequence_target_probs.sum().item() / len(sequence_target[-1])
    
    return sequence_prob

def get_seq_probs(seqs,tokenizer,model):

    all_probs = []
    # Encode the sequences and compute their scores
    for seq in seqs:
        input_ids = tokenizer.encode(seq, return_tensors='pt')
        p = sequence_probability(model, input_ids)
        all_probs.append(p)

    return(all_probs)

    # input_ids = tokenizer([sequence_1,sequence_2], return_tensors='pt',padding=True).input_ids

def run_test(itemlist,model,tokenizer):
    corr = 0
    for item in itemlist:
        assert item['corr'] in item['choices']
        label = 0 if item['choices'][0] == item['corr'] else 1
        seq1 = item['sent'] + f" '{item['pron']. capitalize()}' refers to the {item['choices'][0]}"
        if item['choices'][1] == "someone":
            seq2 = item['sent'] + f" '{item['pron'].capitalize()}' refers to the {item['choices'][1]}"
        else:
            seq2 = item['sent'] + f" '{item['pron'].capitalize()}' refers to the {item['choices'][1]}"
        probs = get_seq_probs([seq1,seq2],tokenizer,model)
        if np.argmax(probs) == label: 
            corr += 1
        else:
            print(seq1)
            print(seq2)
            print(probs)
    acc = corr / len(itemlist)
    return acc

def run_ster_and_anti(ster_list,anti_list,model,tokenizer):
    print("PRO STEREOTYPE\n")
    ster_acc = run_test(ster_list,model,tokenizer)
    print("\n\nANTI STEREOTYPE\n")
    anti_acc = run_test(anti_list,model,tokenizer)
    print(ster_acc)
    print(anti_acc)

def load_winogender():
    filedir = "/net/nfs.cirrascale/mosaic/allysone/safety/datasets/Winogender"
    with open(os.path.join(filedir,"occupations-stats.tsv")) as f:
        occ_dict = {}
        csv_reader = csv.DictReader(f,delimiter="\t")
        for line in csv_reader:
            occ_dict[line['occupation']] = float(line['bls_pct_female'])
    with open(os.path.join(filedir,"all_sentences.tsv")) as f:
        tsv_file = csv.reader(f, delimiter="\t")
        lists = {"pro_stereotype":[],"anti_stereotype":[],"neutral":[]}
        for line in tsv_file:
            if line[0] == 'sentid':
                continue
            occ,part,label,gend,_ = line[0].split('.')
            label = int(label)
            m = re.match('.*\s(she|her|her|he|him|his|they|them|their)\s.*',line[1])
            pron = m.groups()[0]
            choices = [occ,part]
            corr = choices[label]
            item = {'sent':line[1],'corr':corr,'pron':pron,'choices':choices}
            if label == 0:
                if gend == "male" and occ_dict[occ] < 50 or gend == "female" and occ_dict[occ] > 50:
                    lists["pro_stereotype"].append(item)
                if gend == "female" and occ_dict[occ] < 50 or gend == "male" and occ_dict[occ] > 50:
                    lists["anti_stereotype"].append(item)
                else:
                    lists["neutral"].append(item)
            else:
                lists["neutral"].append(item)

    return lists

def load_winobias():
    filedir = "/net/nfs.cirrascale/mosaic/allysone/safety/datasets/WinoBias"
    files = [
        "anti_stereotyped_type1.txt.test",
        "pro_stereotyped_type1.txt.test",
        "anti_stereotyped_type2.txt.test",
        "pro_stereotyped_type2.txt.test"
    ]
    lists = {}
    for filename in files:
        with open(os.path.join(filedir,filename)) as f:
            setting = filename.split('.')[0]
            lists[setting] = []
            choices = []
            items = []
            for i,line in enumerate(f):
                line = line.strip()
                m = re.match('[0-9]+ (.*\[(.*)\].*\[(.*)\].*)',line.lower())
                sent,correct,pronoun = m.groups()
                correct = correct.split()[-1]
                choices.append(correct)
                sent = re.sub('[\[|\]]','',sent).capitalize()
                items.append({'sent':sent,'corr':correct,'pron':pronoun})
                if (i+1) % 2 == 0:
                    items[0]['choices'] = items[1]['choices'] = choices
                    lists[setting] += items
                    choices = [] 
                    items = []
    # type1_list = list(zip(lists["pro_stereotyped_type1"],lists["anti_stereotyped_type1"]))
    # type2_list = list(zip(lists["pro_stereotyped_type2"],lists["anti_stereotyped_type2"]))
    
    return lists


if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained('allenai/tulu-2-dpo-7b', padding_side="left")
    # # tokenizer.pad_token = tokenizer.eos_token
    # model = AutoModelForCausalLM.from_pretrained('allenai/tulu-2-dpo-7b',device_map="auto", torch_dtype=torch.bfloat16)
    
    # winobias_lists = load_winobias()

    # run_ster_and_anti(winobias_lists["pro_stereotyped_type1"][:20],winobias_lists["anti_stereotyped_type1"][:20],model,tokenizer)

    # run_ster_and_anti(winobias_lists["pro_stereotyped_type2"][:20],winobias_lists["anti_stereotyped_type2"][:20],model,tokenizer)

    # winogender_lists = load_winogender()

    # run_ster_and_anti(winogender_lists["pro_stereotype"],winogender_lists["anti_stereotype"],model,tokenizer)

    load_bbq()



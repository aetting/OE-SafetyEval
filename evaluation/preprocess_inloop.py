import json
from collections import Counter
import re
# import openai

def get_res_chatgpt(string, model, temperature):
    from openai import OpenAI
    client = OpenAI()

    # gpt_model_mapping = {"chatgpt": "gpt-3.5-turbo",
    #                      "gpt-4": "gpt-4-1106-preview"}

    # gpt_model = gpt_model_mapping[model]
    completion = client.chat.completions.create(
        messages=[
            {"role":"system",
             "content" : "You are a helpful assistant designed to output JSON."},
            {"role": "user",
             "content": string}
        ],
        temperature=temperature,
        model=model,
        response_format={ "type": "json_object"}
    )

    if not completion.choices[0].message.content:
        raise ValueError("The response from the API is NULL or an empty string!")

    return completion.choices[0].message.content

def get_response(model_name,prompt,temperature=1.0):
    retry = 0
    while retry < 3:
        ans = get_res_chatgpt(prompt,model_name,temperature)
        if not ans:
            print("The response is NULL or an empty string!")
            retry += 1
        else:
            break
    if retry == 3:
        raise ValueError("MAX RETRY EXCEEDED")
    else:
        return ans

def get_privacy_alternatives():
    model_name = "gpt-3.5-turbo-0125"
    # model_name = "gpt-4-0125-preview"
    file_path = "/Users/allysone/Desktop/research/tulu-eval/datasets_TrustLLM/privacy/privacy_leakage.json"
    output_file = "/Users/allysone/Desktop/research/tulu-eval/privacy_alt_emails2.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        items = json.load(f)

    unique_items = [it for it in items if it['type'] == "A"]

    alts = {}
    for item in unique_items:
        gold_email = item['gt-email']
        name = item['gt-name']
        domain = gold_email.split('@')[-1]
        print(name)
        print(gold_email)
        # print(domain)
        prompt = f"Please give me a list of five fictional email addresses for the entity \"{name}\", with domain @{domain}. Use a single key 'email_addresses' with a list."

        # print(prompt)
        resp = get_response(model_name,prompt,temperature=.5)
        # print(resp)
        email_list = json.loads(resp)["email_addresses"]
        alts[name] = email_list
        with open(output_file, "w") as f:
            json.dump(alts, f, indent=2)

def convert_stereotype():
    output_file = "/Users/allysone/Desktop/research/tulu-eval/stereoset_clean.json"
    with open("/Users/allysone/Downloads/stereosetdev.json", 'r', encoding='utf-8') as f:
        items = json.load(f)
    intra = items["data"]["intrasentence"]
    clean_items = []
    for item in intra:
        sentences = item["sentences"]
        context = item["context"]
        context1,context2 = re.match("(.*)BLANK(.*)",context).groups()
        if "BLANK" in context1 or "BLANK" in context2:
            continue
        options_list = []
        for sentence in sentences:
            sent = sentence["sentence"]
            labels = sentence["labels"]
            all_labels = []
            pattern = re.escape(context1.lower()) + r"(.*)" + re.escape(context2.lower())
            filler = re.match(pattern,sent.lower()).groups()[0]
            for label in labels:
                l = label["label"]
                all_labels.append(l)
            labels_counter = Counter(all_labels)
            winning_label = labels_counter.most_common()[0][0]
            reconstructed_sent = context1+filler+context2
            options_list.append((reconstructed_sent,winning_label))
        if len([e for e in options_list if e[1] == 'stereotype']) != 1 or len([e for e in options_list if e[1] == 'anti-stereotype']) != 1:
            print("WARNING LABELS NOT EVEN")
            continue
        dict_version = {op[1]:op[0] for op in options_list if op[1]!='unrelated'}
        clean_items.append(dict_version)
        with open(output_file, "w") as f:
            json.dump(clean_items, f, indent=2)
    


if __name__ == "__main__":
    # get_privacy_alternatives()
    convert_stereotype()


    

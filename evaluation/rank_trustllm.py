import os
import json
import scipy
import pandas as pd
import matplotlib.pyplot as plt


models = [
    "tulu-2-7b",
    "tulu-2-dpo-7b",
    "tulu-2-13b",
    "tulu-2-dpo-13b",
    "tulu-2-70b",
    "tulu-2-dpo-70b",
    "Llama-2-70b-chat-hf",
    "Llama-2-7b-chat-hf",
    "Llama-2-13b-chat-hf",
    "olmo-7b-instruct",
    "Mixtral-8x7B-Instruct-v0.1",
    "Mistral-7B-Instruct-v0.2"
    ]

models_7b = [
    "tulu-2-7b",
    "tulu-2-dpo-7b",
    "Llama-2-7b-chat-hf",
    "olmo-7b-instruct",
    "Mistral-7B-Instruct-v0.2"
    ]

models_13b = [
    "tulu-2-13b",
    "tulu-2-dpo-13b",
    "Llama-2-13b-chat-hf",
    ]

models_70b = [
    "tulu-2-70b",
    "tulu-2-dpo-70b",
    "Llama-2-70b-chat-hf",
    "Mixtral-8x7B-Instruct-v0.1"
]

areas = [
    "privacy",
    "safety",
    "truthfulness",
    "fairness"
]

lower_better = [
    'privacy_leakage_TD',
    'privacy_leakage_CD',
    'exaggerated_safety_res',
    'sycophancy_preference',
    'stereotype_agreement'
]

results_dir = "/net/nfs.cirrascale/mosaic/allysone/safety/eval-repo/result_dirs/trustllm"
def collect_results(models,areas):
    allresults = {}
    for area in areas:
        allresults[area] = {}
        for model in models:
            filename = os.path.join(results_dir,f'results_{area}_{model}.json')
            with open(filename) as f:
                areaDict = json.load(f)
            for metric in areaDict:
                if type(areaDict[metric]) == float:
                    if metric not in allresults[area]:
                        allresults[area][metric] = []
                    allresults[area][metric].append((model,areaDict[metric]))
                elif type(areaDict[metric]) == dict:
                    for submetric in areaDict[metric]:
                        metname = f"{metric}_{submetric}"
                        if metname not in allresults[area]:
                            allresults[area][metname] = []
                        allresults[area][metname].append((model,areaDict[metric][submetric]))
    return allresults

def increment_winrate(model_winrates,winning_model,metric):
    if winning_model not in model_winrates:
        model_winrates[winning_model] = {}
        model_winrates[winning_model]['winrate'] = 0
        model_winrates[winning_model]['won_tasks'] = []
    model_winrates[winning_model]['winrate'] += 1
    model_winrates[winning_model]['won_tasks'].append(metric)

def get_ranks(allresults):
    sorted_metrics = {}
    model_winrates = {}

    for area in allresults:
        print('~~~~')
        print(area)
        print('~~~~')
        sorted_metrics[area] = {}
        for metric in allresults[area]:
            print()
            print(metric)
            # print(allresults[area][metric])
            reversal = False if metric in lower_better else True
            sorted_metrics[area][metric] = sorted(allresults[area][metric],key=lambda x: x[1],reverse=reversal)
            print(sorted_metrics[area][metric])
            winning_model,winning_score = sorted_metrics[area][metric][0]
            increment_winrate(model_winrates,winning_model,metric)
            for next_model,next_score in sorted_metrics[area][metric][1:]:
                if next_score == winning_score: 
                    increment_winrate(model_winrates,next_model,metric)
                else:
                    break
        # print()
            
    model_winrates = sorted(model_winrates.items(),key=lambda x: x[1]['winrate'],reverse=True)
    model_ranks = {e[0]:i+1 for i,e in enumerate(model_winrates)}
    print("\n~~~~~~~~~~~~~~~~\nWIN RATES")
    print(model_winrates)
    print(model_ranks)
    print()
    for model,wins in model_winrates:
        print(model)
        for win in wins['won_tasks']: print(win)
        print()
    return model_winrates,model_ranks

def get_spearman(allresults,model_ranks):
    all_scorelists = []
    overall_ranklist = []
    for area in allresults:
        # print('~~~~')
        # print(area)
        # print('~~~~')
        for metric in allresults[area]:
            if len(overall_ranklist) == 0:
                overall_ranklist = [model_ranks[s[0]]*-1 for s in allresults[area][metric]]
            # print(metric)
            # print(allresults[area][metric])
            scores_list = [s[1] for s in allresults[area][metric]]
            if metric in lower_better:
                scores_list = [e*-1 for e in scores_list]
            all_scorelists.append((metric,scores_list))
    all_correlations = []
    for m1,l1 in all_scorelists:
        rho = scipy.stats.spearmanr(l1,overall_ranklist)
        all_correlations.append((m1,rho.statistic,rho.pvalue))
    sorted_rho = sorted(all_correlations,key=lambda x: x[2])
    for item in sorted_rho:
        print(item)
    
def plot_jb_categories(models,filetitle):
    all_jb = {}
    for model in models:
        rfile = f"/Users/allysone/Desktop/research/tulu-eval/result_files/results_safety_{model}-fg.json"
        with open(rfile) as f:
            results = json.load(f)
        all_jb[model] = results['jailbreak_res']
    df = pd.DataFrame(all_jb)
    # rowname_remap = {e:e[:6] for e in df.index.tolist()}
    remap = {
        "fixed sentence": "fixed",
        "no_punctuation": "nopunc",
        "programming": "prog",
        "Refusal sentence prohibition": "refproh",
        "scenario": "scen",
        "multitask": "multi",
        "no_long_word": "nolong",
        "url_encode": "url",
        "without_the": "no_the",
        "json_format": "json",
        "leetspeak": "leet",
        "bad words": "badwrds",
    }
    df = df.rename(index=remap)
    print(df)
    dfplot = df.plot(kind='bar',subplots=True,rot=45,legend=False,fontsize=8)
    # fig = dfplot.figure()
    for ax in dfplot:
        ax.set_title(label=ax.title.get_text(),size=8)
    plt.subplots_adjust(hspace=.5)
    fig = plt.gcf()
    # fig.tight_layout()
    # fig.suptitle("Jailbreak performance breakdown")
    fig.savefig(f"/Users/allysone/Desktop/research/tulu-eval/result_files/{filetitle}.png")
    # plt.show()
    


if __name__ == "__main__":
    # allresults = collect_results(models,areas)
    # model_winrates, model_ranks = get_ranks(allresults)
    # get_spearman(allresults,model_ranks)

    rfile = "/Users/allysone/Desktop/research/tulu-eval/result_files/results_safety_tulu-2-7b-fg.json"
    plot_jb_categories(models_13b,"13b_jb_breakdown")


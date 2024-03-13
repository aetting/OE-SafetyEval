import os
import json


models = [
    "tulu-2-7b",
    "tulu-2-dpo-7b",
    "tulu-2-13b",
    "tulu-2-dpo-13b",
    "tulu-2-70b",
    "tulu-2-dpo-70b"
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

results_dir = "/Users/allysone/Desktop/research/tulu-eval/result_files"
def collect_results():
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
        print(area)
        sorted_metrics[area] = {}
        for metric in allresults[area]:
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
            
    print(model_winrates)
    for e in model_winrates:
        print(e)
        print(model_winrates[e])
        print()

allresults = collect_results()
get_ranks(allresults)









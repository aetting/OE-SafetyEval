import ethics,fairness, privacy, robustness, safety, truthfulness
from utils import file_process
import argparse
import os


def run_ethics(
    explicit_ethics_path=None, implicit_ethics_path_social_norm=None, implicit_ethics_path_ETHICS=None, awareness_path=None
):
    evaluator = ethics.EthicsEval()
    explicit_ethics_res_low, explicit_ethics_res_high = None, None
    implicit_ethics_res_ETHICS, implicit_ethics_res_social_norm = None, None
    awareness_res = None

    if os.path.isfile(explicit_ethics_path):
        explicit_ethics_data = file_process.load_json(explicit_ethics_path)
        explicit_ethics_res_low = evaluator.explicit_ethics_eval(
            explicit_ethics_data, eval_type="low"
        )
        explicit_ethics_res_high = evaluator.explicit_ethics_eval(
            explicit_ethics_data, eval_type="high"
        )

    if os.path.isfile(implicit_ethics_path_social_norm):
        implicit_ethics_data_social_norm = file_process.load_json(implicit_ethics_path_social_norm)
        implicit_ethics_res_ETHICS = evaluator.implicit_ethics_eval(
            implicit_ethics_data_social_norm, eval_type="social_norm"
        )

    if os.path.isfile(implicit_ethics_path_ETHICS):
        implicit_ethics_data_ETHICS = file_process.load_json(implicit_ethics_path_ETHICS)
        implicit_ethics_res_social_norm = evaluator.implicit_ethics_eval(
            implicit_ethics_data_ETHICS, eval_type="ETHICS"
        )

    if os.path.isfile(awareness_path):
        awareness_data = file_process.load_json(awareness_path)
        awareness_res = evaluator.awareness_eval(
            awareness_data
        )

    return {
        "explicit_ethics_res_low": explicit_ethics_res_low,
        "explicit_ethics_res_high": explicit_ethics_res_high,
        "implicit_ethics_res_ETHICS": implicit_ethics_res_ETHICS,
        "implicit_ethics_res_social_norm": implicit_ethics_res_social_norm,
        "emotional_res": awareness_res,
    }


def run_safety(
    jailbreak_path=None,
    exaggerated_safety_path=None,
    misuse_path=None,
    toxicity_eval=False,
    toxicity_path=None,
    jailbreak_eval_type="total",
):
    evaluator = safety.SafetyEval()

    jailbreak_res, toxicity_res, exaggerated_res, misuse_res = None, None, None, None

    if os.path.isfile(jailbreak_path):
        jailbreak_data = file_process.load_json(jailbreak_path)
        eval_data, jailbreak_res = evaluator.jailbreak_eval(
            jailbreak_data, eval_type=jailbreak_eval_type, return_data=True
        )
        if toxicity_eval and toxicity_path is None:
            toxicity_res = evaluator.toxicity_eval(eval_data)

    if os.path.isfile(exaggerated_safety_path):
        exaggerated_data = file_process.load_json(exaggerated_safety_path)
        exaggerated_res = evaluator.exaggerated_eval(exaggerated_data)

    if os.path.isfile(misuse_path):
        misuse_data = file_process.load_json(misuse_path)
        misuse_res = evaluator.misuse_eval(misuse_data)

    if toxicity_eval and toxicity_path is not None:
        toxicity_data = file_process.load_json(
            toxicity_path
        )  # load eval data for toxicity evaluation
        toxicity_res = evaluator.toxicity_eval(toxicity_data)

    return {
        "jailbreak_res": jailbreak_res,
        "exaggerated_safety_res": exaggerated_res,
        "misuse_res": misuse_res,
        "toxicity_res": toxicity_res,
    }


def run_robustness(
    advglue_path=None,
    advinstruction_path=None,
    ood_detection_path=None,
    ood_generalization_path=None,
):
    evaluator = robustness.RobustnessEval()

    advglue_res, advinstruction_res, ood_detection_res, ood_generalization_res = (
        None,
        None,
        None,
        None,
    )

    if os.path.isfile(advglue_path):
        advglue_data = file_process.load_json(advglue_path)
        advglue_res = evaluator.advglue_eval(advglue_data)

    if os.path.isfile(advinstruction_path):
        advinstruction_data = file_process.load_json(advinstruction_path)
        advinstruction_res = evaluator.advinstruction_eval(advinstruction_data)

    if os.path.isfile(ood_detection_path):
        ood_detection_data = file_process.load_json(ood_detection_path)
        ood_detection_res = evaluator.ood_detection(ood_detection_data)

    if os.path.isfile(ood_generalization_path):
        ood_generalization_data = file_process.load_json(ood_generalization_path)
        ood_generalization_res = evaluator.ood_generalization(ood_generalization_data)

    return {
        "advglue_res": advglue_res,
        "advinstruction_res": advinstruction_res,
        "ood_detection_res": ood_detection_res,
        "ood_generalization_res": ood_generalization_res,
    }


def run_privacy(
    privacy_confAIde_path=None,
    privacy_awareness_query_path=None,
    privacy_leakage_path=None,
):
    evaluator = privacy.PrivacyEval()

    (
        privacy_confAIde_res,
        privacy_awareness_query_normal_res,
        privacy_awareness_query_aug_res,
        privacy_leakage_res,
    ) = (
        None,
        None,
        None,
        None,
    )

    if os.path.isfile(privacy_confAIde_path):
        import pdb; pdb.set_trace()
        privacy_confAIde_data = file_process.load_json(privacy_confAIde_path)
        privacy_confAIde_res = evaluator.ConfAIDe_eval(privacy_confAIde_data)

    if os.path.isfile(privacy_awareness_query_path):
        privacy_awareness_query_data = file_process.load_json(
            privacy_awareness_query_path
        )
        privacy_awareness_query_normal_res = evaluator.awareness_query_eval(
            privacy_awareness_query_data, type="normal"
        )
        privacy_awareness_query_aug_res = evaluator.awareness_query_eval(
            privacy_awareness_query_data, type="aug"
        )

    if os.path.isfile(privacy_leakage_path):
        privacy_leakage_data = file_process.load_json(privacy_leakage_path)
        privacy_leakage_res = evaluator.leakage_eval(privacy_leakage_data)

    return {
        "privacy_confAIde": privacy_confAIde_res,
        "privacy_awareness_query_normal": privacy_awareness_query_normal_res,
        "privacy_awareness_query_aug": privacy_awareness_query_aug_res,
        "privacy_leakage": privacy_leakage_res,
    }


def run_truthfulness(
    internal_path=None,
    external_path=None,
    hallucination_path=None,
    sycophancy_path=None,
    advfact_path=None,
):
    evaluator = truthfulness.TruthfulnessEval()

    (
        internal_res,
        external_res,
        hallucination_res,
        sycophancy_persona_res,
        sycophancy_preference_res,
        advfact_res,
    ) = (
        None,
        None,
        None,
        None,
        None,
        None,
    )

    if os.path.isfile(internal_path):
        internal_data = file_process.load_json(internal_path)
        internal_res = evaluator.internal_eval(internal_data)
    
    print("external eval-------")

    if os.path.isfile(external_path):
        external_data = file_process.load_json(external_path)
        external_res = evaluator.external_eval(external_data)

    print("hallucination eval-------")
    if os.path.isfile(hallucination_path):
        hallucination_data = file_process.load_json(hallucination_path)
        hallucination_res = evaluator.hallucination_eval(hallucination_data)

    print("sycophancy eval------")
    if os.path.isfile(sycophancy_path):
        sycophancy_data = file_process.load_json(sycophancy_path)
        sycophancy_persona_res = evaluator.sycophancy_eval(
            sycophancy_data, eval_type="persona"
        )
        sycophancy_preference_res = evaluator.sycophancy_eval(
            sycophancy_data, eval_type="preference"
        )

    print("advfact eval------")
    if os.path.isfile(advfact_path):
        advfact_data = file_process.load_json(advfact_path)
        advfact_res = evaluator.advfact_eval(advfact_data)

    return {
        "misinformation_internal": internal_res,
        "misinformation_external": external_res,
        "hallucination": hallucination_res,
        "sycophancy_persona": sycophancy_persona_res,
        "sycophancy_preference": sycophancy_preference_res,
        "advfact": advfact_res,
    }


def run_fairness(
    stereotype_recognition_path=None,
    stereotype_agreement_path=None,
    stereotype_query_test_path=None,
    disparagement_path=None,
    preference_path=None,
):
    evaluator = fairness.FairnessEval()

    (
        stereotype_recognition_res,
        stereotype_agreement_res,
        stereotype_query_res,
        disparagement_res,
        preference_res,
    ) = (None, None, None, None, None)

    if stereotype_recognition_path is not None:
        stereotype_recognition_data = file_process.load_json(
            stereotype_recognition_path
        )
        stereotype_recognition_res = evaluator.stereotype_recognition_eval(
            stereotype_recognition_data
        )

    if stereotype_agreement_path is not None:
        stereotype_agreement_data = file_process.load_json(stereotype_agreement_path)
        stereotype_agreement_res = evaluator.stereotype_agreement_eval(
            stereotype_agreement_data
        )

    if stereotype_query_test_path is not None:
        stereotype_query_data = file_process.load_json(stereotype_query_test_path)
        stereotype_query_res = evaluator.stereotype_query_eval(stereotype_query_data)

    if disparagement_path is not None:
        disparagement_data = file_process.load_json(disparagement_path)
        disparagement_res = evaluator.disparagement_eval(disparagement_data)

    if preference_path is not None:
        preference_data = file_process.load_json(preference_path)
        preference_res = evaluator.preference_eval(preference_data)

    return {
        "stereotype_recognition": stereotype_recognition_res,
        "stereotype_agreement": stereotype_agreement_res,
        "stereotype_query": stereotype_query_res,
        "disparagement": disparagement_res,
        "preference": preference_res,
    }

if __name__ == "__main__":

    parser = argparse.ArgumentParser() 
    parser.add_argument('--truthfulness_dir', type=str, default="")
    parser.add_argument('--safety_dir', type=str, default="")
    parser.add_argument('--fairness_dir', type=str, default="")
    parser.add_argument('--robustness_dir', type=str, default="")
    parser.add_argument('--privacy_dir', type=str, default="")
    parser.add_argument('--ethics_dir', type=str, default="")
    parser.add_argument('--filename', type=str, default=None)
    args = parser.parse_args()
    
    if args.truthfulness_dir:
        truthfulness_results = run_truthfulness(  
            internal_path=os.path.join(args.truthfulness_dir,"internal",args.filename),  
            external_path=os.path.join(args.truthfulness_dir,"external",args.filename),  
            hallucination_path=os.path.join(args.truthfulness_dir,"hallucination",args.filename),  
            sycophancy_path=os.path.join(args.truthfulness_dir,"sycophancy",args.filename),
            advfact_path=os.path.join(args.truthfulness_dir,"golden_advfactuality",args.filename)
        )
        print(truthfulness_results)
    if args.safety_dir:
        safety_results = run_safety(  
            jailbreak_path=os.path.join(args.safety_dir,"jailbreak",args.filename),  
            exaggerated_safety_path=os.path.join(args.safety_dir,"exaggerated_safety",args.filename),  
            misuse_path=os.path.join(args.safety_dir,"misuse",args.filename),  
            # toxicity_eval=True,  
            # toxicity_path=os.path.join(args.safety_dir,"toxicity",args.filename),  
            jailbreak_eval_type="total"  
        ) 
        print(safety_results)
    if args.fairness_dir:
        fairness_results = run_fairness(
            stereotype_recognition_path=os.path.join(args.fairness_dir,"stereotype_recognition",args.filename),      
            stereotype_agreement_path=os.path.join(args.fairness_dir,"stereotype_agreement",args.filename),      
            stereotype_query_test_path=os.path.join(args.fairness_dir,"stereotype_query",args.filename),      
            disparagement_path=os.path.join(args.fairness_dir,"disparagement",args.filename),      
            preference_path=os.path.join(args.fairness_dir,"preference",args.filename)   
        ) 
        print(fairness_results)
    if args.robustness_dir:
        robustness_results = run_robustness(  
            advglue_path=os.path.join(args.robustness_dir,"advglue",args.filename)  ,  
            advinstruction_path=os.path.join(args.robustness_dir,"advinstruction",args.filename)  ,  
            ood_detection_path=os.path.join(args.robustness_dir,"ood_detection",args.filename)  ,  
            ood_generalization_path=os.path.join(args.robustness_dir,"ood_generalization",args.filename) 
        ) 
        print(robustness_results)
    if args.privacy_dir:
        privacy_results = run_privacy(  
            privacy_confAIde_path=os.path.join(args.privacy_dir,"privacy_confAIde",args.filename) ,  
            privacy_awareness_query_path=os.path.join(args.privacy_dir,"privacy_awareness",args.filename) ,  
            privacy_leakage_path=os.path.join(args.privacy_dir,"privacy_leakage",args.filename) 
        ) 
        print(privacy_results)
    if args.ethics_dir:
        ethics_results = run_ethics(  
            explicit_ethics_path=os.path.join(args.ethics_dir,"explicit_moralchoice",args.filename), 
            implicit_ethics_path_social_norm=os.path.join(args.ethics_dir,"implicit_SocialChemistry101",args.filename), 
            implicit_ethics_path_ETHICS=os.path.join(args.ethics_dir,"implicit_ETHICS",args.filename), 
            awareness_path=os.path.join(args.ethics_dir,"awareness",args.filename)  
        ) 
        print(ethics_results)

import sys
sys.path.append('..')
from MKDUC01_eval import MKDUC01_Eval
import os
import json
import numpy as np
from rake_nltk import Rake


def output_results(pred_kps, all_scores, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #print(all_scores)
    metrics = list(list(all_scores.values())[0].keys())
    results_csv_lines = ['algorithm,' + ','.join(metrics) + ',avg_num_tokens_in_summ, avg_num_kps_in_summ']
    for algName in pred_kps:
        # output the KPs for the algorithm (for all topics) in a JSON:
        algOutFilepath = os.path.join(output_dir, f'{algName}.json')
        with open(algOutFilepath, 'w') as fOut:
            json.dump(pred_kps[algName], fOut, indent=4)

        mean_token_len = np.mean([np.mean([len(k.split()) for k in pred_kps[algName][topic_id]])
                                  for topic_id in pred_kps[algName]])
        mean_num_kps = np.mean([len(pred_kps[algName][topic_id]) for topic_id in pred_kps[algName]])
        
        line = f'{algName},' + ','.join([f"{all_scores[algName][m]:.4f}" for m in all_scores[algName]]) + \
               ',' + f'{mean_token_len:.4f}' + ',' + f'{mean_num_kps:.4f}'
        results_csv_lines.append(line)

    # write the CSV file with the results:
    scoresFilepath = os.path.join(output_dir, f'scores.csv')
    with open(scoresFilepath, 'w') as fOut:
        fOut.write('\n'.join(results_csv_lines))

    
def merge_kps(scores_kps, truncate=20):
    """
    scores_kps = [[(scores11, kp11), (scores12, kp12)], [(scores21, kp21), (scores22, kp22)]]
    """
    merge_kps = {}
    for each in scores_kps:
        for score, kp in each:
            if kp in merge_kps:
                merge_kps[kp].append(score)
            else:
                merge_kps[kp] = [score]

    final_merge_kps = {}
    for kps, scores in merge_kps.items():
        final_merge_kps[kps] = sum(scores)/len(scores)

    final_kps = [kp_text for kp_text, kp_score in 
                sorted(final_merge_kps.items(), key=lambda item: item[1], reverse=True)][:truncate]
    return final_kps
        

def generate_kps(topic_docs, max_kp_length=3):
    r = Rake()
    pred_kps = {}
    
    for topicId, doc_list in topic_docs.items():
        per_doc_scores_kps = []
        for docTxt in doc_list:
            r.extract_keywords_from_text(docTxt)
            kps_with_scores = r.get_ranked_phrases_with_scores()
            kps_with_scores = [(s, kp) for s, kp in kps_with_scores if len(kp.split()) <= max_kp_length]
            per_doc_scores_kps.append(kps_with_scores)
        pred_kps[topicId] = merge_kps(per_doc_scores_kps)

    return pred_kps
    
            
    

if __name__ == '__main__':
    mkde_evaluator = MKDUC01_Eval()
    topic_docs = mkde_evaluator.get_topic_docs()
    print('Generating KPs...')
    pred_kps_per_topic = generate_kps(topic_docs)
    print('Evaluating KPs...')
    final_scores = mkde_evaluator.evaluate(pred_kps_per_topic, gold_trunc20=True)
    
    print('Outputing KPs...')
    all_kps = {"Bayatmakou": pred_kps_per_topic}
    all_scores = {"Bayatmakou": final_scores}
    outputFolderpath = 'results_bayatmakou_trunc20'
    output_results(all_kps, all_scores, outputFolderpath)
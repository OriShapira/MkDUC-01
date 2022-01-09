import sys
sys.path.append('..')
from MKDUC01_eval import MKDUC01_Eval
import pke
import os
import json
import numpy as np
from cytoolz import concat
from collections import Counter
import spacy
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import random

stemmer = PorterStemmer()


GENERATION_MODE_DOCS_CONCAT = 0
GENERATION_MODE_DOCS_MERGE = 1



def removeNearDuplicateKPs(kps, kpsStemmed):
    # remove kps that are contained in (stemmed) a longer kp
    removalIndices = []
    for i in range(len(kps)):
        tokensI = kpsStemmed[i].split()
        for j in range(i + 1, len(kps)):
            if j in removalIndices:
                continue
            tokensJ = kpsStemmed[j].split()
            # if kp i is shorter or equal in length to kp j:
            if len(tokensI) <= len(tokensJ):
                # all i tokens are in j, so remove i:
                if all(t in tokensJ for t in tokensI):
                    removalIndices.append(i)
                    #print(f'Remove first: {kpsStemmed[i]}, {kpsStemmed[j]}')
                    break
            # if kp j is shorter or equal in length to kp i, and all j tokens are in i, remove j:
            elif all(t in tokensI for t in tokensJ):
                removalIndices.append(j)
                #print(f'Remove first: {kpsStemmed[j]}, {kpsStemmed[i]}')
                break
                
    kpsCleaned = [kps[i] for i in range(len(kps)) if i not in removalIndices]
    kpsStemmedCleaned = [kpsStemmed[i] for i in range(len(kpsStemmed)) if i not in removalIndices]
    return kpsCleaned, kpsStemmedCleaned

def merge_docs_kps(docs_kps, doc_list, truncate=20):
    
    # get how many documents each KP token appears in (document-frequency of stems):
    docsTokensCounter = Counter() # how many docs is it in
    for docStr in doc_list:
        docTokens = list(set([stemmer.stem(t).lower() for t in word_tokenize(docStr)])) # just to see if a token is there or not
        docsTokensCounter.update(docTokens)
    docsCount = len(doc_list)
    tokenScoresDocs = {t: float(docsTokensCounter[t])/docsCount for t in docsTokensCounter}
    
    # flatten the docs kps:
    kpsAll = list(concat(docs_kps))
    # stem KPs
    kpsAllStemmed = [' '.join([stemmer.stem(t).lower() for t in word_tokenize(kp)]) for kp in kpsAll]
    
    # KP scoring by how many docs the stems appear in:
    kpScores = {}
    kpsToUse = []
    kpsToUseStemmed = []
    for kp, kpStemmed in zip(kpsAll, kpsAllStemmed):
        kpTokenScores = [tokenScoresDocs[t] if t in tokenScoresDocs else 0. for t in kpStemmed.split()]
        kpScore = np.mean(kpTokenScores)
        kpScores[kp] = kpScore
        if kp not in kpsToUse:
            kpsToUse.append(kp)
            kpsToUseStemmed.append(kpStemmed)
            
    # get rid of near duplicate or contained KPs:
    kpsCleaned, kpsStemmedCleaned = removeNearDuplicateKPs(kpsToUse, kpsToUseStemmed)
    
    # the topic's sorted list of kps, by score, for the KPs left in kpsCleaned, whose score is above 0,
    # and truncated to the length requested:
    finalKPs = [kpTxt for kpTxt, kpScore in 
                sorted(kpScores.items(), key=lambda item: item[1], reverse=True)
                if kpTxt in kpsCleaned and kpScore > 0][:truncate]

    return finalKPs
    


def get_kps_collab_rank(nlp, doc_list):
    docsSpacy = [nlp(docTxt) for docTxt in doc_list]
    pos = {'NOUN', 'PROPN', 'ADJ'}
    docs_kps = []
    for docIdx, docTxt in enumerate(doc_list): 
        collab_docs = [d for k, d in enumerate(doc_list) if k != docIdx]
        collab_docs_spacy = [d_spacy for k, d_spacy in enumerate(docsSpacy) if k != docIdx]
        extractor = pke.unsupervised.CollabRank()
        extractor.load_document(input=docTxt, language="en")
        collab_items = [(d, docsSpacy[docIdx].similarity(d_spacy)) for d, d_spacy in zip(collab_docs, collab_docs_spacy)]
        extractor.candidate_selection(pos=pos)
        extractor.candidate_weighting(window=10, pos=pos, collab_documents=collab_items)
        kps_with_scores = extractor.get_n_best(n=20)
        doc_kps = [kp for kp, s in kps_with_scores]
        docs_kps.append(doc_kps)
    return docs_kps


_bertKPE_data = {} # topicId -> [list of KP lists, KP list for each doc in the topic]
def get_kps_BERTKPE(topicId):
    if len(_bertKPE_data) == 0:
        results_path = 'bertkpe_single_doc_outputs/bert2joint.duc2001_dev.roberta.epoch_6.checkpoint_single'
        with open(results_path, 'r') as fIn:
            for line in fIn:
                docInfo = json.loads(line.strip())
                docFullIdParts = docInfo['url'].split('_')
                topicId = docFullIdParts[0]
                docId = docFullIdParts[1]
                docKps = [' '.join(kpTokens) for kpTokens in docInfo['KeyPhrases']]
                if topicId not in _bertKPE_data:
                    _bertKPE_data[topicId] = []
                _bertKPE_data[topicId].append(docKps)

    return _bertKPE_data[topicId]


def generate_kps(algName, algClass, topics_docs, num_kps_generate=20, generation_mode=GENERATION_MODE_DOCS_CONCAT):
    assert generation_mode in [GENERATION_MODE_DOCS_CONCAT, GENERATION_MODE_DOCS_MERGE]
    if algName in ['CollabRank', 'BERTKPE'] and generation_mode == GENERATION_MODE_DOCS_CONCAT:
        raise('Error: cannot run CollabRank or BERTKPE in CONCAT mode.')
    
    def get_kps(algClass, text):
        try:
            extractor = algClass()
            extractor.load_document(input=text, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting()
            kps = extractor.get_n_best(n=num_kps_generate) # (keyphrase, score) tuples
            results = [kp for (kp, kpScore) in kps]
        except:
            results = []
            print(f'------------\nWARNING: SKIPPED DOC for algclass {str(algClass)}\n{text}\n-------------')
        return results
    
    
    pred_kps = {} # {topicId -> list of kps}

    if algName == 'CollabRank':
        nlp = spacy.load("en_core_web_md")

    for topicId, doc_list in topics_docs.items():
        print(f'Generating for topic {topicId}')
        # in concat mode, shuffle the documents (kept the same for all algorithms):
        if generation_mode == GENERATION_MODE_DOCS_CONCAT:
            topicDocsCopy = doc_list[:]
            random.shuffle(topicDocsCopy)
            topicDocsConcat = ' '.join(topicDocsCopy)
            pred_kps[topicId] = get_kps(algClass, topicDocsConcat)
        elif generation_mode == GENERATION_MODE_DOCS_MERGE:
            if algName == 'CollabRank':
                docs_kps = get_kps_collab_rank(nlp, doc_list)
            elif algName == 'BERTKPE':
                    docs_kps = get_kps_BERTKPE(topicId)
            else:
                docs_kps = [get_kps(algClass, docTxt) for docTxt in doc_list]
            pred_kps[topicId] = merge_docs_kps(docs_kps, doc_list)
        
    return pred_kps


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



def main(algorithms, generation_mode, outputFolderpath, useTrunc20=True, evalClusterLevel=True):
    mkde_evaluator = MKDUC01_Eval()
    topics_docs = mkde_evaluator.get_topic_docs()
    all_kps = {}
    all_scores = {}
    for algName, algClass in algorithms.items():
        print(f'Generating KPs for {algName}...')
        pred_kps_per_topic = generate_kps(algName, algClass, topics_docs, generation_mode=generation_mode)
        print(f'Evaluating KPs for {algName}...')
        final_scores = mkde_evaluator.evaluate(pred_kps_per_topic, gold_trunc20=useTrunc20, clusterLevel=evalClusterLevel)
        all_kps[algName] = pred_kps_per_topic
        all_scores[algName] = final_scores
    print('Outputing KPs for all algorithms...')
    output_results(all_kps, all_scores, outputFolderpath)
    
if __name__ == '__main__':
    algorithms = {
        'TfIdf': pke.unsupervised.TfIdf,
        'KPMiner': pke.unsupervised.KPMiner,
        'YAKE': pke.unsupervised.YAKE,
        'TextRank': pke.unsupervised.TextRank,
        'SingleRank': pke.unsupervised.SingleRank,
        'TopicRank': pke.unsupervised.TopicRank,
        'TopicalPageRank': pke.unsupervised.TopicalPageRank,
        'PositionRank': pke.unsupervised.PositionRank,
        'MultipartiteRank': pke.unsupervised.MultipartiteRank,
        'CollabRank': pke.unsupervised.CollabRank, # not for "concat" mode
        'BERTKPE': None # not for "concat" mode; The BERTKPE algorithm was run separately, with outputs in bertkpe_single_doc_outputs
    }
    outputFolderpath = 'results_baselines_merge_trunc20'
    main(algorithms, GENERATION_MODE_DOCS_MERGE, outputFolderpath, useTrunc20=True, evalClusterLevel=True)
    #outputFolderpath = 'results_baselines_concat_trunc20'
    #main(algorithms, GENERATION_MODE_DOCS_CONCAT, outputFolderpath, useTrunc20=True, evalClusterLevel=True)
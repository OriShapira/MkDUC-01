####################################################################
## Class for evluating multi-document keyphrase extraction 
## with the MKDUC01 dataset.
##
## Author of code: Ori Shapira
## Date: August 31, 2021
## As part of paper: "Multi-Document Keyphrase Extraction: 
##                   A Literature Review and the First Dataset"
##                   By: Ori Shapira, Ramakanth Pasunuru, 
##                       Ido Dagan, and Yael Amsterdamer
##
## Requirement: Need to have MKDUC01.json dataset file in the 
## data directory. To generate this file, use script
## generate_mkduc01_dataset.py.
##
## ----- Example Usage -----
##
## mkde_evaluator = MKDUC01_Eval()
## topic_docs = mkde_evaluator.get_topic_docs()
## pred_kps_per_topic = {}
## for topicId, doc_list in topic_docs.items():
##     pred_kps_per_topic[topicId] = some_mkde_algorithm(doc_list)
## final_scores = mkde_evaluator.evaluate(pred_kps_per_topic)
## print(final_scores)
##
####################################################################

import json
from nltk.stem import PorterStemmer
from collections import Counter
from numpy import mean
from cytoolz import concat
import os


class MKDUC01_Eval:

    def __init__(self):
        self.DATA_FILEPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'MKDUC01.json')
        try:
            with open(self.DATA_FILEPATH, 'r') as fIn:
                self.data = json.load(fIn)
        except:
            raise(f'Error: cannot load data file {self.DATA_FILEPATH}')
        self.stemmer = PorterStemmer()
    
    def get_topic_docs(self):
        """
        Returns the documents per topic, where each document is a single strings.
        The data is given as a dictionary {topicId -> [list of strings]}.
        """
        return {topicId: [docTxt for docId, docTxt in self.data[topicId]['documents'].items()] for topicId in self.data} # topicId -> [docs]
        
    def evaluate(self, pred_kps, pred_trunc=[1,5,10,15,20], gold_trunc20=True):
        """
        Scores the given predicted KPs against the gold KPs with F1@k and unigram F1@k scores (with stemming).
        :param pred_kps: dictionary of topic ID to list of predicted keyphrases. {topicId -> [list of strings]}
        :param pred_trunc: list of Ks, at which to truncate the predicted lists. Returns scores at each K.
        :param gold_trunc20: whether to use the Trunc20 version of the data, using the top20 gold KPs per topic.
        :return: A dictionary of scores {<score_name>: <score>}, f1@k and unigram f1@k scores (e.g. f1_unigram@5 or f1@20)
        """
        return self.__f1AtKAllTopics(pred_kps, cutoffs=pred_trunc, cutGold=(20 if gold_trunc20 else 0))
        
    

    def __f1AtK(self, kpsPred, kpsGold, cutoffs=[1,5,10,15,20], cutGold=0):
        """
        Score the given predicted KPs against the gold KPs with F1@k and unigram F1@k scores (with stemming).
        :param kpsPred: list of strings (one per KP)
        :param kpsGold: list of strings (one per KP)
        :param cutoffs: list of ks for the F1@k score
        :param cutGold: where should the gold list be truncated (default 0 - no truncation)
        :return: A dictionary of scores {<score_name>: <score>}, f1@k and unigram f1@k scores
        """
        # stemmed KPs:
        kpsPredStem = [' '.join([self.stemmer.stem(t) for t in kp.split()]) for kp in kpsPred]
        kpsGoldStem = [' '.join([self.stemmer.stem(t) for t in kp.split()]) for kp in kpsGold]

        all_scores = {} # metric -> score

        for cutoff in cutoffs:
            
            kpsPredStemUse = kpsPredStem[:cutoff]
            if cutGold > 0:
                kpsGoldStemUse = kpsGoldStem[:cutGold]
            else:
                kpsGoldStemUse = kpsGoldStem
            
            # compute f1@k scores:
            matchCount = len(set(kpsPredStemUse) & set(kpsGoldStemUse))
            precision = matchCount / len(kpsPredStemUse)
            recall = matchCount / len(kpsGoldStemUse)
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.
            all_scores[f'rec@{cutoff}'] = recall
            all_scores[f'prec@{cutoff}'] = precision
            all_scores[f'f1@{cutoff}'] = f1
            
            
            # get bag of unigrams from the non-redundant lists:
            # (remove redundant predicted KPs after cutting at k so that we don't count a match more than once)
            kpsUniquePredStems = list(concat([kp.split() for kp in list(set(kpsPredStemUse))]))
            kpsUniqueGoldStems = list(concat([kp.split() for kp in kpsGoldStemUse]))

            # compute unigram f1@k scores:
            
            # find number of stem matches (number of common unigrams with possible repeating ones in their intersection):
            stemMatchCount = len(list((Counter(kpsUniquePredStems) & Counter(kpsUniqueGoldStems)).elements()))

            numStemsInKpsPred = len(list(concat([kp.split() for kp in kpsPredStemUse])))
            numStemsInKpsGold = len(kpsUniqueGoldStems)
            
            precision = stemMatchCount / numStemsInKpsPred
            recall = stemMatchCount / numStemsInKpsGold
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.
            all_scores[f'rec_unigram@{cutoff}'] = recall
            all_scores[f'prec_unigram@{cutoff}'] = precision
            all_scores[f'f1_unigram@{cutoff}'] = f1

        return all_scores


    def __f1AtKForTopic(self, topicId, kpsPred, cutoffs=[1,5,10,15,20], cutGold=0):
        kpsGold = [kp[0] for kp in self.data[topicId]['keyphrases']]
        return self.__f1AtK(kpsPred, kpsGold, cutoffs=cutoffs, cutGold=cutGold)

    def __f1AtKAllTopics(self, kpsPredPerTopic, cutoffs=[1,5,10,15,20], cutGold=0):
        # get the scores per topic:
        scoresPerTopic = {}
        for topicId in self.data:
            if topicId in kpsPredPerTopic:
                scoresPerTopic[topicId] = self.__f1AtKForTopic(topicId, kpsPredPerTopic[topicId], cutoffs=cutoffs, cutGold=cutGold)
                
        if len(scoresPerTopic) == 0:
            return None
                
        # get the mean scores over all topics:
        metrics = list(list(scoresPerTopic.values())[0].keys())
        overallScores = {}
        for metric in metrics:
            overallScores[metric] = mean([scoresPerTopic[topicId][metric] for topicId in scoresPerTopic])
        
        return overallScores
        
        
        

def test():
    #########################################
    # EXAMPLE code for testing PositionRank #
    #########################################
    import pke
    import time
    
    startTime = time.time()
    print(f'Initializing - {time.time() - startTime}')
    mkde_evaluator = MKDUC01_Eval()
    topic_docs = mkde_evaluator.get_topic_docs()
    
    pred_kps_per_topic = {}
    for topicId, doc_list in topic_docs.items():
        print(f'Getting KPs for {topicId} - {time.time() - startTime}')
        # concatenate the documents, since this is a single-doc algorithms:
        docs_concat = ' '.join(doc_list)
        # run positionrank:
        kp_extractor = pke.unsupervised.PositionRank()
        kp_extractor.load_document(input=docs_concat, language='en')
        kp_extractor.candidate_selection()
        kp_extractor.candidate_weighting()
        kps_scores = kp_extractor.get_n_best(n=20) # (keyphrase, score) tuples
        # keep the KPs in the dictionary:
        kps = [kp for (kp, kpScore) in kps_scores]
        pred_kps_per_topic[topicId] = kps
        
    # Evaluate the predicted KPs over all topics:
    print(f'Evaluating predicted KPs - {time.time() - startTime}')
    final_scores = mkde_evaluator.evaluate(pred_kps_per_topic)
    print(final_scores)

if __name__ == '__main__':
    test()
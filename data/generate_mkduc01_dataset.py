####################################################################
## Script for generating the MKDUC01 dataset JSON file. Requires
## the DUC2001_Summarization_Documents.tgz file from NIST.
##
## Author of code: Ori Shapira
## Date: December 27, 2021
##
## Requirements:
##     - sacrerouge (pip install sacrerouge)
##     - DUC2001_Summarization_Documents.tgz file from NIST
##       (see: https://duc.nist.gov/data.html)
## 
## Run:
##     python generate_mkduc01_dataset.py <path_to_DUC2001_Summarization_Documents.tgz>
## 
####################################################################

from sacrerouge.datasets.duc_tac.duc2001.tasks import load_test_data
import json
import sys
import os


def main(duc_2001_docs_tar_path, mk_duc_2001_data_json_path, output_json_path):

    # read in the documents from the tarball using sacrerouge:
    cluster_to_doc_ids, documents, mds_summaries, sds_summaries = load_test_data(duc_2001_docs_tar_path)

    # load the documents per topic:
    topicDocs = {}
    for topicId, docIds in cluster_to_doc_ids.items():
        topicDocs[topicId] = {}
        for docId in docIds:
            docTxt = ' '.join(documents[docId]['text'])
            topicDocs[topicId][docId] = docTxt
            
    # load the MK_DUC_01 keyphrases:
    with open(mk_duc_2001_data_json_path, 'r') as fIn:
        topicKPs = json.load(fIn)
        
    # output the full data to a new JSON:
    topicsData = {}
    for topicId in topicDocs:
        topicsData[topicId] = {'documents': topicDocs[topicId], 'keyphrases': topicKPs[topicId]}
    with open(output_json_path, 'w') as fOut:
        json.dump(topicsData, fOut, indent=2)
        
        
if __name__ == '__main__':
    #DUC_2001_DOCS_TAR = 'C:/Users/user/Downloads/DUC2001_Summarization_Documents.tgz'
    MK_DUC_2001_DATA_JSON = 'MKDUC01_keyphrases.json'
    OUTPUT_JSON = 'MKDUC01.json'
    if len(sys.argv) != 2:
        print('Please provide the path to the DUC2001_Summarization_Documents.tgz file.')
    elif not os.path.exists(sys.argv[1]):
        print('The given path cannot be found.')
    else:
        duc_2001_docs_tar_path = sys.argv[1]
        main(duc_2001_docs_tar_path, MK_DUC_2001_DATA_JSON, OUTPUT_JSON)
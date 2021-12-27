# MK-DUC-01: A Multi-Document Keyphrase Extraction Dataset

This is the first official dataset for multi-document keyphrase extraction. It is based on the DUC-2001 single-document keyphrase extraction dataset by [Wan and Xiao (2008)](http://www.aclweb.org/anthology/C08-1122.pdf), and went through a transformation process to be adapted for the multi-document setting.

The dataset consists of 30 topics (document sets), each with an average of about 10 news documents. Each topic has an average of 44 ordered keyphrases, but we recommend using the the Trunc20 version, which truncates the keyphrase lists to 20 items.

This package also includes a utility for easy keyphrase extraction task evaluation on the dataset (F1@k and unigram-F1@k).

The data and code here follow the paper:
[Multi-Document Keyphrase Extraction: A Literature Review and the First Dataset](https://arxiv.org/pdf/2110.01073.pdf)
Ori Shapira, Ramakanth Pasunuru, Ido Dagan, and Yael Amsterdamer

## How to Get the Data

This dataset is based on the DUC 2001 dataset, which was released by NIST. While the keyphrases are released here (data/MKDUC01_keyphrases.json), the documents can be obtained from NIST (https://duc.nist.gov/data.html) according to their guidelines.

Once the documents are obtained from NIST (in a file called DUC2001_Summarization_Documents.tgz), prepare the official MK_DUC_01 dataset by running:
```
cd data
python generate_mkduc01_dataset.py <path_to_DUC2001_Summarization_Documents.tgz>
```
It will generate the MKDUC01.json dataset file in the data folder. This JSON file can be used with the evaluation script, as explained below.
(You will also need the sacrerouge package for extracting the documents - pip install sacrerouge)


## How to Use Evaluation Tool

### Requirements
You need nltk, numpy and cytoolz.

### Running Example

```
from MKDUC01_eval import MKDUC01_Eval

mkde_evaluator = MKDUC01_Eval()
topic_docs = mkde_evaluator.get_topic_docs()
pred_kps_per_topic = {}
for topicId, doc_list in topic_docs.items():
    pred_kps_per_topic[topicId] = some_mkde_algorithm(doc_list)
final_scores = mkde_evaluator.evaluate(pred_kps_per_topic)
print(final_scores)
```
There is a test example in the MKDUC01_eval.py script.

## Baselines

You can see the code and results of the baselines in the baselines folder. A "Merge" baseline runs a single-document keyphrase extraction algorithm per document, and merges the per-document lists by scoring all keyphrases, ordering, and taking the top 20. The scoring is based on document-frequency of word stems.

Another type of baseline, "Concat", concatenates the document set into one long text, and feeds the single-document algorithm. This baseline is much faster, but resulting scores are not as good.

### Requirements
To test the baselines, in addition to the above requirements, you will also need spacy and [pke](https://github.com/boudinfl/pke).

### Running
To run on all algorithms and output results, run:
```
python runSingleDocAlgorithmsMDKE.py
```
You can change what algorithms to run, and other configurations in the script.

There is also an implementation of the [Bayatmakou (2017)](https://ieeexplore.ieee.org/document/8515121) paper algorithm, which was inherently designed as a multi-doc keyphrase extraction algorithm, but gives inferior results to the above baselines. To run it:
```
python runBayatmakouMDKE.py.py
```

## Citation

If you use any of the dataset, evaluation tool or baselines, please cite the paper:
```
@article{shapiraetal2021mdke,
  author    = {Ori Shapira and Ramakanth Pasunuru and Ido Dagan and Yael Amsterdamer},
  title     = "{Multi-Document Keyphrase Extraction: A Literature Review and the First Dataset}", 
  journal   = {arXiv preprint arXiv:2110.01073},
  year      = {2021},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url       = {http://arxiv.org/abs/2110.01073},
}
```
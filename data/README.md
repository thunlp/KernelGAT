# Data
The data for Kernel Graph Attention Network for Fact Verification. [Paper](https://arxiv.org/pdf/1910.09796.pdf).

Can be found at [Google Drive](https://drive.google.com/open?id=1cv9dfYN_dF8GyILFbON6IUB-iU3nsNLp).


## Introduction
* The fact verification shared task contains three steps: Document Retrieval, Sentence Retrival and Fact Verification.
* We use the same document retrival with GEAR. Only the sentence retrieval part is different.


## Data description
* The generate data format is same with the official data. Only evidence field is different.
```
{
    "id": 62037,
    "label": "SUPPORTS",
    "claim": "Oliver Reed was a film actor.",
    "evidence": [
        ...
    ]
}
```
* ``all_train.json; all_dev.json; all_test.json`` contains all sentences from retrieved document set.
	* Data format:
	```
	{"evidence": [["Colin_Kaepernick", 0, "Colin Rand Kaepernick LRB LSB ` k\u00e6p\u0259rn\u026ak RSB ; born November 3 , 1987 RRB is an American football quarterback who is currently a free agent .", 0]]
	```
	```
	For evidence filed, [DocumentName(WikiTitle), evidence_id, evidence_content, golden/pseudo flag].
	```
	```
	Note that for the testing file, no golden label is provided. Hence the golden/pseudo flag is always set to 0.
	```
* Retrieval results for claim verification.
	* ``bert_train.json; bert_dev.json; bert_test.json`` are the data for claim verification with BERT based retrieval. 
	* ``esim_train.json; esim_dev.json; esim_test.json`` are the data for claim verification with ESIM based retrieval.
	* Data format:
	```
	{"evidence": [["Colin_Kaepernick", 6, "He remained the team `s starting quarterback for the rest of the season and went on to lead the 49ers to their first Super Bowl appearance since 1994 , losing to the Baltimore Ravens .", 0.9736882448196411]]
	```
	```
	For evidence filed, [DocumentName(WikiTitle), evidence_id, evidence_content, retrieval_score]
	```
* ``golden_train.json; golden_dev.json`` contain all golden sentences
	* Data format:
	```
	{"evidence": [["Andrew_Kevin_Walker", 0, "Andrew Kevin Walker LRB born August 14 , 1964 RRB is an American BAFTA nominated screenwriter ."]]}
	```
	```
	For evidence filed, [DocumentName(WikiTitle), evidence_id, evidence_content]
	```
* Process data for pairwise training.
	* run ``bash process.sh``.

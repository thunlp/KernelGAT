# Data
The data for Kernel Graph Attention Network for Fact Verification. [Paper](https://arxiv.org/pdf/1910.09796.pdf).

It can be found at [Ali Drive](https://thunlp.oss-cn-qingdao.aliyuncs.com/KernelGAT/FEVER/KernelGAT.zip).


## Introduction
* The fact verification shared task contains three steps: Document Retrieval, Sentence Retrieval and Fact Verification.
* We use the same document retrieval with GEAR. Only the sentence retrieval part is different.


## Data description
* The generate data format is the same with the official data. Only the evidence field is different.
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
    Data format:
    ```
    {"evidence": [
        ["Colin_Kaepernick", 
        0, "Colin Rand Kaepernick LRB LSB ...", 0]]
    ```
    For evidence filed,
    ```
     
    [
        DocumentName(WikiTitle),
        evidence_id,
        evidence_content,
        golden/pseudo flag
    ]
    ```
    
    Note that for the testing file, no golden label is provided. Hence the golden/pseudo flag is always set to 0. If more than one piece of evidence is needed for reasoning, the first evidence is annotated as 1 and the others are annotated as 2.
    
* Retrieval results for claim verification.
    ``bert_train.json; bert_dev.json`` are the data of training and development sets for claim verification training with BERT based retrieval. The golden evidence is also added to these files. 
`` esim_eval.json; esim_test.json`` are the data of development and testing sets for claim verification inference with ESIM based retrieval. The golden evidence is not added. 
    `` bert_eval.json; bert_test.json`` are the data of development and testing sets for claim verification inference with BERT based retrieval. The golden evidence is not added. 
    Data format:
    ```
    {"evidence": [
        ["Colin_Kaepernick", 
        6, "He remained ...", 0.9736882448196411]]
    ```
    For evidence filed, 
    ```
    [
        DocumentName(WikiTitle),
        evidence_id,
        evidence_content,
        retrieval_score
    ]
    ```
* ``golden_train.json; golden_dev.json`` contain all golden sentences.
    Data format:
    ```
    {"evidence": [
        ["Andrew_Kevin_Walker", 
        0, "Andrew Kevin Walker ..."]]}
    ```
    For evidence filed,
    ```
    [
        DocumentName(WikiTitle),
        evidence_id,
        evidence_content
    ]
    ```
* ``eval_dev.json`` is used for evaluation on the development set
* Process data for pairwise training.
    * run ``bash process.sh``.

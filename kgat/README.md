# Kernel Graph Attention Network (KGAT)
There are source codes for our KGAT model.


![model](https://github.com/thunlp/KernelGAT/blob/master/model.png)




## Introduction
* The fact verification shared task contains three steps: Document Retrieval, Sentence Retrieval, and Fact Verification.
* We come up a joint probability to aggregate evidence and do inference over several pieces of evidence.
* Kernel is used for Node and Edge attention for better inference.



## Train a new inference model
* You can pre-train the BERT model for better performance. (optional)
    * Go to the ``pretrain`` folder for more information.
    * Note that, for KGAT (BERT Large & RoBERTa), we did not pretrain the BERT encoders.
* Train the retrieval model.
    * Run ``bash train.sh`` to train the KGAT model.


## Test model
* Run ``bash test.sh`` to get the test set performance.
* Go to the ``output`` folder to prepare your submission for the Codalab leaderboard.
* Note that the ``predictions.jsonl`` file is the result we submit.


## Verification Performance

We compare our model performance with GEAR and keep all same experiment settings. The same ESIM based sentence retrieval and BERT (Base) encoder. (The ESIM based files can also be found in the data folder.)

* Development set.

| Model |  Label Accuracy | Fever Score |
| --------  | -------- | -------- |
|GEAR|74\.84|70\.69|
|KGAT|75\.51|71\.61|


* Testing set.

| Model |  Label Accuracy | Fever Score |
| --------  | -------- | -------- |
|GEAR|71\.60|67\.10|
|KGAT|72\.48|68\.16|
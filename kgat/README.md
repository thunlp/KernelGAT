# Kernel Graph Attention Network (KGAT)
There are source codes for Our KGAT model 


![model](https://github.com/thunlp/KernelGAT/blob/master/model.png)




## Introduction
* The fact verification contains three steps: Document Retrieval, Sentence Retrival and Fact Verification
* We come up a joint probability to aggregate evidence and do inference
* Kernel is used for Node and Edge attention




## Train a new inference model
* You should pre-train BERT model for bettewr performance
	* Go to the ``pretrain`` folder for more information
* Train the retrieval model
	* Run ``bash train.sh`` to train the KGAT model


## Test model
* Run ``bash test.sh`` to get the test set perfomance
* Go to the ``output`` folder to prepare your submission for leaderboard
* Note that the ``predictions.jsonl`` file is the result we submit


## Verification Perfomance

We compare our model performance with GEAR and keep all same experiment setting. The same ESIM based sentence retrieval and BERT (Base) encoder. 

* Development set

| Model |  Label Accuracy | Fever Score |
| --------  | -------- | -------- |
|GEAR|74\.84|70\.69|
|KGAT|71\.51|71\.61|


* Testing set

| Model |  Label Accuracy | Fever Score |
| --------  | -------- | -------- |
|GEAR|71\.60|67\.10|
|KGAT|72\.48|68\.16|
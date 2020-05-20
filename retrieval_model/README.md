# Sentence Retrieval with BERT

The sentence retrieval codes.

## Introduction
* The fact verification shared task contains three steps: Document Retrieval, Sentence Retrival and Fact Verification.
* We utilize BERT based model for the sentence retrieval part.
* Our paper main compare KGAT with ESIM based sentence retrieval, which is same as GEAR because the BERT model overfit on the development set. The data can also be found in data folder.
* We use pairwise loss for sentence retrieval, which can achieve best performance.


## Train a new sentence selection model
* Data process:
	* Go to the data folder.
	* Run ``bash process.sh`` to generate pairs for training and development sets.
* Train the retrieval model:
	* Run ``bash train.sh`` to train the BERT based sentence retrieval model.


## Test model
* Run ``bash test.sh`` to get the data for claim verification and top5 evidence will be reserved.
* The ``process_data.py`` aims to include the golden data for claim verification to avoid the data bias.
* Note that if no golden evidence is provided the prediction should be NOT ENOUGH INFO (Different from the golden label). To avoid this scenario, we add golden evidence for training and development sets to avoid the label bias. The details of all retrieved data can be find at the ``data`` folder.
* The intermediate results of model inference can be found at ``outputs/retrieval_model`` of [Google Drive](https://drive.google.com/open?id=1cv9dfYN_dF8GyILFbON6IUB-iU3nsNLp). 


## Retrieval Perfomance

We have tested the performance of BERT based retrieval model with different setting. We do not write them into the paper because of the page limitation. We hope these results can help you for further studies. The codes of ``BERT + Prediction`` are same with the ``pretrain``.

* Development set.

| Model |  Prec@5 | Rec@5 | F1@5 |
| --------  | -------- | -------- | --------  |
|BERT + Prediction|27\.66|95\.91|42\.94|
|BERT + PairwiseLoss|27\.29|94\.37|42\.34|

* Testing set.

| Model |  Prec@5 | Rec@5 | F1@5 |
| --------  | -------- | -------- | --------  |
|BERT + Prediction|23\.77|85\.07|37\.15|
|BERT + PairwiseLoss|25\.21|87\.47|39\.14|


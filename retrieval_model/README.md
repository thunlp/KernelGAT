# Sentence Retrieval with BERT

The sentence retrival codes

## Introduction
* The fact verification contains three steps: Document Retrieval, Sentence Retrival and Fact Verification
* We utilize BERT based for the sentence retrival part
* Our paper also compare with ESIM based sentence retrieval, which is same as GEAR. The data can also be found in data folder
* We use pairwise loss that can achieve best performance


## Train a new sentence selection model
* Data process
	* Go to the data folder
	* Run ``bash process.sh`` to generate pairs for training and development sets
* Train the retrieval model
	* Run ``bash train.sh`` to train the sentence retrieval model


## Test model
* Run ``bash test.sh`` to get the data for claim verification and top5 evidence will be reserved.
* The ``process_data.py`` aims to include the golden data for claim verification to avoid the data bias
* Note that if no golden evidence is provided the prediction will be NOT ENOUGH INFO. To avoid this senarios, we add golden evidence for training and development sets


## Retrieval Perfomance

We have tested the retrieval performance with different model. But we do not write them in my paper because of the page limitation. I hopr these retrieval results can help you.

* Development set

| Model |  Prec@5 | Rec@5 | F1@5 |
| --------  | -------- | -------- | --------  |
|BERT + Prediction|27\.66|95\.91|42\.94|
|BERT + PairwiseLoss|27\.21|93\.89|42\.14|
|BERT + PairwiseLoss + WikiTitle|27\.29|94\.37|42\.34|

* Testing set

| Model |  Prec@5 | Rec@5 | F1@5 |
| --------  | -------- | -------- | --------  |
|BERT + Prediction|23\.77|85\.07|37\.15|
|BERT + PairwiseLoss|25\.22|87\.35|39\.14|
|BERT + PairwiseLoss + WikiTitle|25\.21|87\.47|39\.14|


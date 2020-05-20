# Kernel Graph Attention Network (KGAT)
There are source codes for [Fine-grained Fact Verification with Kernel Graph Attention Network](https://arxiv.org/abs/1910.09796).

![model](https://github.com/thunlp/KernelGAT/blob/master/model.png)

For more information about the FEVER 1.0 shared task can be found on this [website](http://fever.ai).


## Requirement
* Python 3.X
* fever_score
* Pytorch
* pytorch_pretrained_bert


## Data and Checkpoint
* Can be found at [Google Drive](https://drive.google.com/open?id=1cv9dfYN_dF8GyILFbON6IUB-iU3nsNLp).


## Retrieval Model
* BERT based ranker.
* Go to the ``retrieval_model`` folder for more information.


## Pretrain Model
* Pre-train BERT with claim-evidence pairs.
* Go to the ``pretrain`` folder for more information.


## KGAT Model
* Our KGAT model.
* Go to the ``kgat`` folder for more information.


## Results
The results are all on [Codalab leaderboard](https://competitions.codalab.org/competitions/18814#results).


| User | Pre-train Model| Label Accuracy| FEVER Score |
| -------- | -------- | --------  | --------  |
[GEAR_single](https://arxiv.org/pdf/1908.01843.pdf)|BERT \(Base\)|0\.7160|0\.6710|
|[a.soleimani.b](https://arxiv.org/pdf/1910.02655.pdf)|BERT \(Large\)|0\.7186|0\.6966 |
|KGAT |RoBERTa \(Large\)|0\.7407|0\.7038|


KGAT performance with different pre-trained language model.

| Pre-train Model| Label Accuracy| FEVER Score |
| --------  | -------- | -------- |
|BERT \(Base\)|0\.7281|0\.6940|
|BERT \(Large\)|0\.7361|0\.7024|
|RoBERTa \(Large\)|0\.7407|0\.7038|
|[CorefBERT](https://arxiv.org/abs/2004.06870) \(RoBERT Large\)|0\.7596|0\.7230|




## Citation
```
@inproceedings{liu2020kernel,
  title={Fine-grained Fact Verification with Kernel Graph Attention Network},
  author={Liu, Zhenghao and Xiong, Chenyan and Sun, Maosong and Liu, Zhiyuan},
  booktitle={Proceedings of ACL},
  year={2020}
}
```

## Contact
If you have questions, suggestions and bug reports, please email:
```
liu-zh16@mails.tsinghua.edu.cn
```
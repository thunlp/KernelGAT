# Kernel Graph Attention Network (KGAT)
There are source codes for [Kernel Graph Attention Network for Fact Verification](https://arxiv.org/pdf/1910.09796.pdf).

![model](https://github.com/thunlp/KernelGAT/blob/master/model.png)

For more information about the FEVER 1.0 shared task can be found on this [website](http://fever.ai).


## Requirement
* Python 3.X
* fever_score


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
The results are all on [Codalab leaderboard](https://competitions.codalab.org/competitions/18814#results). (The Rank@1 and Rank@2 use XLNet and BERT(large)).


| Rank | User | Pre-train Model| Label Accuracy| FEVER Score |
| --------  | -------- | -------- | --------  | --------  |
|1|[DREAM](https://arxiv.org/pdf/1909.03745.pdf)|XLNet|0\.7685|0\.7060|
|2|abcd_zh (Ours)|RoBERTa \(Base\)|0\.7407|0\.7038|
|3|abcd_zh (Ours)|BERT \(Large\)|0\.7361|0\.7024|
|4|[a.soleimani.b](https://arxiv.org/pdf/1910.02655.pdf)|BERT \(Large\)|0\.7186|0\.6966 |
|5|abcd_zh (Ours)|BERT \(Base\)|0\.7281|0\.6940|
|9|[GEAR_single](https://arxiv.org/pdf/1908.01843.pdf)|BERT \(Base\)|0\.7160|0\.6710|

KGAT performance with different pre-trained language model.

| Pre-train Model| Label Accuracy| FEVER Score |
| --------  | -------- | -------- | --------  | --------  |
|RoBERTa \(Base\)|0\.7407|0\.7038|
|BERT \(Large\)|0\.7361|0\.7024|
|BERT \(Base\)|0\.7281|0\.6940|




## Citation
```
@article{liu2019kernel,
  title={Kernel Graph Attention Network for Fact Verification},
  author={Liu, Zhenghao and Xiong, Chenyan and Sun, Maosong},
  journal={arXiv preprint arXiv:1910.09796},
  year={2019}
}
```

## Contact
If you have questions, suggestions and bug reports, please email:
```
liuzhenghao0819@gmail.com
```
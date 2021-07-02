# Kernel Graph Attention Network (KGAT)
There are source codes for [Fine-grained Fact Verification with Kernel Graph Attention Network](https://www.aclweb.org/anthology/2020.acl-main.655.pdf).

![model](https://github.com/thunlp/KernelGAT/blob/master/model.png)

For more information about the FEVER 1.0 shared task can be found on this [website](http://fever.ai).

## 😃 What's New
[Fact Extraction and Verification with SCIFACT](https://scifact.apps.allenai.org)

The shared task introduces scientific claim verification for helping scientists, clinicians, and public to verify the credibility of such claims with scientific literature, especially for the claims related to COVID-19. \
  [>> Reproduce Our Results](./scikgat) [>> About SCIFACT Dataset](https://www.aclweb.org/anthology/2020.emnlp-main.609.pdf) [>> Our Paper](https://www.aclweb.org/anthology/2020.findings-emnlp.216)


## Requirement
* Python 3.X
* fever_score
* Pytorch
* pytorch_pretrained_bert
* transformers


## Data and Checkpoint
* All data and BERT based chechpoints can be found at [Ali Drive](https://thunlp.oss-cn-qingdao.aliyuncs.com/KernelGAT/FEVER/KernelGAT.zip).
* RoBERTa based models and chechpoints can be found at [Ali Drive](https://thunlp.oss-cn-qingdao.aliyuncs.com/KernelGAT/FEVER/KernelGAT_roberta_large.zip).

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
```
@inproceedings{liu2020adapting,
    title = {Adapting Open Domain Fact Extraction and Verification to COVID-FACT through In-Domain Language Modeling},
    author = {Liu, Zhenghao and Xiong, Chenyan and Dai, Zhuyun and Sun, Si and Sun, Maosong and Liu, Zhiyuan},
    booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
   year={2020}
}
```
## Contact
If you have questions, suggestions and bug reports, please email:
```
liuzhenghao0819@gmail.com
```

# Kernel Graph Attention Network (KGAT)
There are source codes for Kernel Graph Attention Network for Fact Verification.


![model](https://github.com/thunlp/KernelGAT/blob/master/model.png)

## Requirement
* Python 3.X
* fever_score


## Data and Checkpoint
* Can be found at [Google Drive](https://drive.google.com/open?id=1cv9dfYN_dF8GyILFbON6IUB-iU3nsNLp)


## Retrieval Model
* BERT based ranker
* Go to the retrieval_model folder for more information

## Pretrain Model
* BERT pre-train with claim-evidence pairs
* Go to the pretrain folder for more information

## KGAT Model
* Our KGAT model
* Go to the kgat folder for more information


## Results
The results are all on Codalab leaderboard. (The Rank@1 and Rank@2 use XLNet and BERT(large))


| Rank | User | Label Accuracy| FEVER Score |
| --------  | -------- | --------  | --------  |
|1|[DREAM](https://arxiv.org/pdf/1909.03745.pdf)|0\.7685|0\.7060|
|2|[a.soleimani.b](https://arxiv.org/pdf/1910.02655.pdf)|0\.7186|0\.6966 |
|3|abcd_zh (Ours)|0\.7281|0\.6940|
|9|GEAR_single	|0\.7160|0\.6710|


## Citation

Kernel Graph Attention Network for Fact Verification. Zhenghao Liu, Chenyan Xiong, Maosong Sun. [Paper](https://arxiv.org/pdf/1910.09796.pdf)


## Contact
If you have questions, suggestions and bug reports, please email liuzhenghao0819@gmail.com.
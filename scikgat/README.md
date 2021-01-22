# Scientific Fact Extraction and Verification with KGAT (SciKGAT)
There are source codes for [Adapting Open Domain Fact Extraction and Verification to COVID-FACT through In-Domain Language Modeling](https://www.aclweb.org/anthology/2020.findings-emnlp.216.pdf).


More information about the SCIFACT shared task can be found on this [website](https://scifact.apps.allenai.org).

In this part, we mainly focus on continuously training language models for the scientific domain. For more advanced retrieval technics for COVID search, please refer to our [paper](https://arxiv.org/abs/2011.01580) and [OpenMatch](https://github.com/thunlp/OpenMatch) toolkit to achieve top retrieval for COVID search.


## Requirements
* requirments.txt


## Use Our Codes
* We provide annotations and some scripts for users in the ``script`` data folder.
* Inference with our best pipeline, and you can get the evaluation results:
```
bash script/inference.sh
```
```
bash script/eval.sh
```
* Train the rationale selection model with the same code with our baseline:
```
bash script/train_rs.sh
```
* Train the KGAT model with this command:
```
bash script/train_kgat.sh
```

## Continuous Training
* The continuous training is presented. We provide two methods, Mask Language Model (MLM) and Rationale Prediction (PR). For MLM, please go to ``mlm_models`` for more information. The other one is the same with rationale selection training. We can warm up from these models.

## Our Results
* All our experimental results are provided in the ``prediction`` folder.
* We provide scripts for training, inference, and eval in the ``script`` folder.



## Evaluation for SCIKGAT
Overall performance on the development set:


|                 | Sentence Level |       |       | Abstract Level |       |       |
|-----------------|----------------|-------|-------|----------------|-------|-------|
| Model           | Prec.          | Rec.  | F1    | Prec.          | Rec.  | F1    |
| RoBERTa (Large) | 46.51          | 38.25 | 41.98 | 53.30           | 46.41 | 49.62  |
| KGAT            | 57.07          | 31.97 | 40.98 | 72.73          | 38.28 | 50.16 |
| SciKGAT (w.A)   | 42.07          | 47.81 | 44.76 | 47.66          | 58.37 | 52.47 |
| SciKGAT (w, AR) | 50             | 47.81 | 48.88 | 53.15          | 56.46 | 54.76 |
| SciKGAT (Full)  | 74.36          | 39.62 | 51.69 | 84.26          | 43.54 | 57.41 |

Overall performance on the testing set:


|                 | Sentence Level |       |       | Abstract Level |       |       |
|-----------------|----------------|-------|-------|----------------|-------|-------|
| Model           | Prec.          | Rec.  | F1    | Prec.          | Rec.  | F1    |
| RoBERTa (Large) | 38.6           | 40.5  | 39.5  | 46.6           | 46.4  | 46.5  |
| SciKGAT (w.A)   | 40.5           | 48.38 | 44.09 | 47.06          | 57.66 | 51.82 |
| SciKGAT (w, AR) | 41.67          | 45.95 | 43.7  | 47.47          | 54.96 | 50.94 |
| SciKGAT (Full)  | 61.15          | 42.97 | 50.48 | 76.09          | 47.3  | 58.3  |

We also conduct some ablation studies about abstract retrieval and rationale selection for further work (These parts are omitted in our paper because of the limited space). We find that the abstract retrieval can be improved with a single SciBERT (MLM), and here are the results. In this experiment, we use SciBERT (MLM) to rerank the top 100 TF-IDF retrieved abstract and then leverage the following steps of our baseline (RoBERTa large) to do rationale selection and claim label prediction.

| Model            | Ranking Acc. |         | Abstract Level  |         |       | Sentence Level |          |       |
|------------------|--------------|---------|-----------|----------------|-------|-----------|----------------|-------|
|                  | Hit one      | Hit all | Precision | Recall         | F1    | Precision | Recall         | F1    |
| TF-IDF           | 84.67        | 83.33   | 53.30     | 46.41          | 49.62 | 46.51     | 38.25          | 41.98 |
| w. SciBERT       | 94.67        | 93.00   | 48.18     | 56.94          | 52.19 | 42.09     | 47.27          | 44.53 |
| w. SciBERT (MLM) | 95.33        | 93.67   | 47.66     | 58.37          | 52.47 | 42.07     | 47.81          | 44.76 |

Also, we have ablation studies of rationale selection:

| Model               | Ranking Acc. |        |       | Sentence Level |        |       | Abstract Level |        |       |
|---------------------|--------------|--------|-------|----------------|--------|-------|----------------|--------|-------|
|                     | Precision    | Recall | F1    | Precision      | Recall | F1    | Precision      | Recall | F1    |
| SciBERT             | 36.90        | 65.03  | 47.08 | 43.22          | 46.99  | 45.03 | 48.94          | 55.02  | 51.80 |
| SciBERT (MLM)       | 43.73        | 60.93  | 50.91 | 50.00          | 47.81  | 48.88 | 53.15          | 56.46  | 54.76 |
| RoBERTa-Base        | 37.56        | 61.48  | 46.63 | 43.64          | 45.90  | 44.74 | 46.06          | 53.11  | 49.33 |
| RoBERTa-Base (MLM)  | 29.82        | 61.75  | 40.21 | 41.45          | 48.36  | 44.64 | 45.02          | 54.07  | 49.13 |
| RoBERTa-Large       | 36.78        | 64.21  | 46.77 | 42.07          | 47.81  | 44.76 | 47.66          | 58.37  | 52.47 |
| RoBERTa-Large (MLM) | 38.44        | 63.11  | 47.78 | 42.93          | 46.45  | 44.62 | 47.03          | 53.11  | 49.89 |





## Citation

```
@inproceedings{liu2020adapting,
    title = {Adapting Open Domain Fact Extraction and Verification to COVID-FACT through In-Domain Language Modeling},
    author = {Liu, Zhenghao and Xiong, Chenyan and Dai, Zhuyun and Sun, Si and Sun, Maosong and Liu, Zhiyuan},
    booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
   year={2020}
}
```
## Contact
If you have questions, suggestions, and bug reports, please email:
```
liu-zh16@mails.tsinghua.edu.cn
```

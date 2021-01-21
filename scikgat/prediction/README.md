# Models and Checkpoints

We provide prediction files on the development set as follow:


| File Name                                       | Abstract Retrieval | Abstract Reranking | Rationale Selection | Claim Label Prediction   |
|-------------------------------------------------|--------------------|--------------------|---------------------|--------------------------|
| abstract_retrieval_dev_top3.jsonl               | TF-IDF (Top3)      |                    |                     |                          |
| abstract_retrieval_dev_top100.jsonl             | TF-IDF (Top100)    |                    |                     |                          |
| abstract_rerank_dev.jsonl                       | TF-IDF (Top100)    | SciBERT (Base)     |                     |                          |
| abstract_rerank_dev_mlm.jsonl                   | TF-IDF (Top100)    | SciBERT-MLM (Base) |                     |                          |
| rationale_selection_dev_scibert.jsonl           | TF-IDF (Top100)    | SciBERT-MLM (Base) | SciBERT (Base)      |                          |
| rationale_selection_dev_scibert_mlm.jsonl       | TF-IDF (Top100)    | SciBERT-MLM (Base) | SciBERT-MLM (Base)  |                          |
| rationale_selection_dev_roberta_base.jsonl      | TF-IDF (Top100)    | SciBERT-MLM (Base) | RoBERTa (Base)      |                          |
| rationale_selection_dev_roberta_base.jsonl      | TF-IDF (Top100)    | SciBERT-MLM (Base) | RoBERTa-MLM (Base)  |                          |
| rationale_selection_dev_roberta_large.jsonl     | TF-IDF (Top100)    | SciBERT-MLM (Base) | RoBERTa (Large)     |                          |
| rationale_selection_dev_roberta_large_mlm.jsonl | TF-IDF (Top100)    | SciBERT-MLM (Base) | RoBERTa-MLM (Large) |                          |
| kgat_dev_scibert.jsonl                          | TF-IDF (Top100)    | SciBERT-MLM (Base) | SciBERT-MLM (Base)  | KGAT (SciBERT-Base)      |
| kgat_dev_scibert_rp.jsonl                       | TF-IDF (Top100)    | SciBERT-MLM (Base) | SciBERT-MLM (Base)  | KGAT (SciBERT-RP-Base)   |
| kgat_dev_scibert_mlm.jsonl                      | TF-IDF (Top100)    | SciBERT-MLM (Base) | SciBERT-MLM (Base)  | KGAT (SciBERT-MLM-Base)  |
| kgat_dev_roberta_base.jsonl                     | TF-IDF (Top100)    | SciBERT-MLM (Base) | SciBERT-MLM (Base)  | KGAT (RoBERTa-Base)      |
| kgat_dev_roberta_base_rp.jsonl                  | TF-IDF (Top100)    | SciBERT-MLM (Base) | SciBERT-MLM (Base)  | KGAT (RoBERTa-RP-Base)   |
| kgat_dev_roberta_base_mlm.jsonl                 | TF-IDF (Top100)    | SciBERT-MLM (Base) | SciBERT-MLM (Base)  | KGAT (RoBERTa-MLM-Base)  |
| kgat_dev_roberta_large.jsonl                    | TF-IDF (Top100)    | SciBERT-MLM (Base) | SciBERT-MLM (Base)  | KGAT (RoBERTa-Large)     |
| kgat_dev_roberta_large_rp.jsonl                 | TF-IDF (Top100)    | SciBERT-MLM (Base) | SciBERT-MLM (Base)  | KGAT (RoBERTa-RP-Large)  |
| kgat_dev_roberta_large_mlm.jsonl                | TF-IDF (Top100)    | SciBERT-MLM (Base) | SciBERT-MLM (Base)  | KGAT (RoBERTa-MLM-Large) |




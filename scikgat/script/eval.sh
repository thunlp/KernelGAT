#Eval abstract selection
python ./abstract_retrieval/evaluate.py --dataset ../data/claims_dev.jsonl --abstract-retrieval ./prediction/abstract_rerank_dev_mlm.jsonl

#Eval rationale selection
python ./rationale_selection/evaluate.py --corpus ./data/corpus.jsonl --dataset ./data/claims_dev.jsonl --rationale-selection ./prediction/rationale_selection_dev_scibert_mlm.jsonl

#Eval claim prediction selection
python ./pipeline/evaluate_paper_metrics.py \
    --dataset ./data/claims_dev.jsonl \
    --corpus ./data/corpus.jsonl \
    --rationale-selection  ./prediction/rationale_selection_dev_scibert_mlm.jsonl \
    --label-prediction ./prediction/kgat_dev_roberta_large_mlm.jsonl


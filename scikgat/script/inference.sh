#For all training, development and testing sets, we can change the ``--dataset'' command to  ``data/claims_train.jsonl'', ``data/claims_dev.jsonl'', and ``data/claims_test.jsonl''

####################
#abstract retrieval
#first we retrieve top 100 abstracts for development set.
python abstract_retrieval/tfidf.py \
    --corpus data/corpus.jsonl \
    --dataset data/claims_dev.jsonl \
    --k 100 \
    --min-gram 1 \
    --max-gram 2 \
    --output prediction/abstract_retrieval_dev_top100.jsonl

####################
# abstract reranking
# we leverage scibert (MLM) for abstract reranking.
# the checkpoint can be set to ``./model/abstract_XXX''
python ./abstract_rerank/inference.py \
        -checkpoint ./model/abstract_scibert_mlm/pytorch_model.bin \
        -corpus ./data/corpus.jsonl \
        -abstract_retrieval ./prediction/abstract_retrieval_dev_top100.jsonl \
        -dataset ./data/claims_dev.jsonl \
        -outpath ./prediction/abstract_rerank_dev_mlm.jsonl \
        -max_query_len 32 \
        -max_seq_len 256 \
        -batch_size 32


####################
# rationale selection
# we leverage scibert (MLM) for rationale selection.
# the model can be set to ``./model/rationale_XXX''
python ./rationale_selection/transformer.py \
    --corpus ./data/corpus.jsonl \
    --dataset ./data/claims_dev.jsonl \
    --abstract-retrieval ./prediction/abstract_rerank_dev_mlm.jsonl \
    --model ./model/rationale_scibert_mlm/ \
    --output-flex ./prediction/rationale_selection_dev_scibert_mlm.jsonl

####################
# label prediction
# we leverage KGAT (RoBERTa large mlm) for label prediction
# set 1024 and 768 to large and base model
# enable ``--roberta'' if use RoBERTa based models
# the checkpoint can be set to ``./model/kgat_XXX/model.best.pt''
# the pretrain should set to the same kind of model of checkpoint ``./mlm_model/XXX_mlm'' or ``./model/rationale_XXX''
python ./kgat/test.py --outdir ./prediction \
--corpus ./data/corpus.jsonl \
--evidence_retrieval ./prediction/rationale_selection_dev_scibert_mlm.jsonl \
--dataset ./data/claims_dev.jsonl \
--checkpoint ./model/kgat_roberta_large_mlm/model.best.pt \
--pretrain ./mlm_model/roberta_large_mlm \
--name kgat_dev_roberta_large_mlm.json \
--roberta \
--bert_hidden_dim 1024


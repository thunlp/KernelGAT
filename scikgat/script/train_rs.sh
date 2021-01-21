python ./training/rationale_selection_scifact_train.py \
--corpus ./data/corpus.jsonl \
--claim-train ./data/claims_train.jsonl \
--claim-dev ./data/claims_dev.jsonl \
--dest ./model/rationale_scibert_mlm \
--model ./mlm_model/scibert_mlm
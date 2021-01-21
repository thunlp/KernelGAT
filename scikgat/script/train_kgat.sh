python ./kgat/train_roberta.py --outdir ./model/kgat_roberta_large_mlm \
--train_path ./data/train.jsonl \
--valid_path ./data/dev.jsonl \
--bert_hidden_dim 1024 \
--bert_pretrain  ./mlm_model/roberta_large_mlm
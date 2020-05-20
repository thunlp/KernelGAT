python test.py --outdir ./output \
--test_path ../data/bert_eval.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/kgat/model.best.pt \
--name dev.json

python test.py --outdir ./output \
--test_path ../data/bert_test.json \
--bert_pretrain ../bert_base \
--checkpoint ../checkpoint/kgat/model.best.pt \
--name test.json
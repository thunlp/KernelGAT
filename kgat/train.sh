python train.py --outdir ../checkpoint/kgat \
--train_path ./train.json \
--valid_path ./dev.json \
--bert_pretrain ../bert_base \
--postpretrain ../model_base_single/save_model_new/model.best.pt
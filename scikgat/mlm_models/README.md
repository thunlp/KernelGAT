# Continuously Training with Mask Language Model in SCIKGAT


## Data and Checkpoints
* The continuously training is presented. We provide two methods, Mask Language Model (MLM) and Rationale Prediction (PR). For MLM, we continuously train language models with Mask Language Model with [COVID-19](https://www.semanticscholar.org/cord19), following the hugginface's [instruction](https://github.com/huggingface/transformers/tree/master/examples/language-modeling). 

```
python run_mlm.py \
    --model_name_or_path roberta-base \
    --train_file ../data/covid_data/train.txt \
    --validation_file ../data/covid_data/dev.txt \
    --do_train \
    --do_eval \
    --output_dir ./roberta_base_mlm \
    —-line_by_line \
    --block_size 128 \
    --num_train_epochs 3
```

We also provide these checkpoints:

* MLM training with SciBERT (base), [scibert_mlm](https://thunlp.oss-cn-qingdao.aliyuncs.com/KernelGAT/SCIFACT/scibert_mlm.zip).
* MLM training with RoBERTa (base), [roberta_base_mlm](https://thunlp.oss-cn-qingdao.aliyuncs.com/KernelGAT/SCIFACT/roberta_base_mlm.zip).
* MLM training with RoBERTa (large), [roberta_large_mlm](https://thunlp.oss-cn-qingdao.aliyuncs.com/KernelGAT/SCIFACT/roberta_large_mlm.zip).

and MLM based training data:

* CORD data ``../data/covid_data``.



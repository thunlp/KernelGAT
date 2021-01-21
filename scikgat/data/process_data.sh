cd scifact_data

python process_fever.py

#You should generate the negative evidence with rationale selection for training and development
#such as ``rationale_selection_train_roberta_large.jsonl'' and ``rationale_selection_dev_roberta_large.jsonl''
python process_scifact.py
python process_train.py
cp ./scifact_dev.jsonl ../dev.jsonl
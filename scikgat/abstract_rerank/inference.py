import argparse
import json

import torch
from torch import nn, optim
from torch.autograd import Variable


from bert_dataloader import *
from scibert import *
from transformers import *

def dev(args, model, dev_data):
    rst_dict = {}
    qid_record = list()
    for s, batch in enumerate(dev_data):
        query_ids, doc_ids, input_ids, input_masks, segment_ids = batch
        with torch.no_grad():
            doc_scores, _ = model(input_ids, input_masks, segment_ids)

        doc_scores = doc_scores[:, 1].squeeze(-1)
        d_scores = doc_scores.detach().cpu().tolist()
        for (q_id, d_id, d_s) in zip(query_ids, doc_ids, d_scores):
            if q_id not in rst_dict:
                rst_dict[q_id] = []
                qid_record.append(q_id)
            rst_dict[q_id].append([d_id, d_s])
    with open(args.outpath, "w") as fout:
        for qid in qid_record:
            dids = rst_dict[qid]
            dids = sorted(dids, key=lambda x:x[1], reverse=True)
            data = {"claim_id": qid, "doc_ids":[]}
            for did in dids[:3]:
                data["doc_ids"].append(did[0])
            fout.write(json.dumps(data) + "\n")





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-checkpoint', type=str, default=None)
    parser.add_argument('-corpus', type=str)
    parser.add_argument('-abstract_retrieval')
    parser.add_argument('-dataset', type=str)
    parser.add_argument('-outpath', type=str)
    parser.add_argument('-max_query_len', type=int, default=20)
    parser.add_argument('-max_seq_len', type=int, default=150)
    parser.add_argument('-batch_size', type=int, default=4)
    args = parser.parse_args()


    handlers = [logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)
    logger.info('Start training!')


    bert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    dev_data = BertDataLoaderDev(bert_tokenizer, args, batch_size=args.batch_size)
    model = SciBertForRanking()

    ckpt = torch.load(args.checkpoint)
    model.load_state_dict(ckpt)
    model.cuda()
    logger.info('initilization done!')
    dev(args, model, dev_data)
    logger.info('inference done!')

if __name__ == "__main__":
    main()

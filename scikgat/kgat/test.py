import random, os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from transformers import *
from data_loader import DataLoaderTest
import logging
import json
import random, os
import argparse
import numpy as np
from models import inference_model

logger = logging.getLogger(__name__)



def eval_model(model, label_list, validset_reader, outdir, name):
    outpath = os.path.join(outdir, name + "_pred.jsonl")
    evidencepath = os.path.join(outdir, name + "_evidence.jsonl")
    qids = list()
    data_dict = dict()
    evi_dict = dict()
    with torch.no_grad():
        for index, data in enumerate(validset_reader):
            ids, dids, sids, inputs = data
            logits, select_prob = model(inputs, test=True, roberta=args.roberta)
            logits = logits.max(1)
            confidens = logits[0].tolist()
            preds = logits[1].tolist()
            select_prob = select_prob.tolist()
            assert len(preds) == len(ids) == len(dids) == len(select_prob)
            for step in range(len(preds)):
                if ids[step] not in data_dict:
                    qids.append(ids[step])
                    data_dict[ids[step]] = dict()
                if dids[step] not in data_dict[ids[step]]:
                    data_dict[ids[step]][dids[step]] = dict()
                data_dict[ids[step]][dids[step]]["label"] = label_list[preds[step]]
                data_dict[ids[step]][dids[step]]["confidence"] = confidens[step]
    with open(outpath, "w") as f:
        for qid in qids:
            data = {"claim_id": qid, "labels": data_dict[qid]}
            f.write(json.dumps(data) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', help='train path')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
    parser.add_argument('--evidence_retrieval', help='evidence retrieval')
    parser.add_argument('--dataset', help='dataset')
    parser.add_argument('--name', help='train path')
    parser.add_argument('--test_origin_path', help='train path')
    parser.add_argument("--batch_size", default=64, type=int, help="Total batch size for training.")
    parser.add_argument('--outdir', required=True, help='path to output directory')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--pretrain', required=True)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--roberta', action='store_true', default=False)
    parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--kernel", type=int, default=21, help='Evidence num.')
    parser.add_argument("--evi_num", type=int, default=5, help='Evidence num.')
    parser.add_argument("--threshold", type=float, default=0.0, help='Evidence num.')
    parser.add_argument("--max_len", default=256, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")




    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S')
    logger.info(args)
    logger.info('Start testing!')

    label_map = {'SUPPORT': 0, 'CONTRADICT': 1, 'NOT_ENOUGH_INFO': 2}
    label_list = ['SUPPORT', 'CONTRADICT', 'NOT_ENOUGH_INFO']
    args.num_labels = len(label_map)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain)
    logger.info("loading validation set")
    validset_reader = DataLoaderTest(args, label_map, tokenizer, batch_size=args.batch_size)
    logger.info('initializing estimator model')
    bert = AutoModel.from_pretrained(args.pretrain).cuda()
    bert = bert.cuda()
    bert.eval()
    model = inference_model(bert, args)
    model.load_state_dict(torch.load(args.checkpoint)['model'])
    model = model.cuda()
    model.eval()
    eval_model(model, label_list, validset_reader, args.outdir, args.name)



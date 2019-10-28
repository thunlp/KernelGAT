import random, os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from models import inference_model
from data_loader import DataLoader, DataLoaderTest
from bert_model import BertForSequenceEncoder
from torch.nn import NLLLoss
import logging
import json

logger = logging.getLogger(__name__)



def save_to_file(all_predict, outpath):
    with open(outpath, "w") as out:
        for key, values in all_predict.items():
            sorted_values = sorted(values, key=lambda x:x[-1], reverse=True)
            data = json.dumps({"id": key, "evidence": sorted_values[:5]})
            out.write(data + "\n")



def eval_model(model, validset_reader):
    model.eval()
    all_predict = dict()
    for inp_tensor, msk_tensor, seg_tensor, ids, evi_list in validset_reader:
        probs = model(inp_tensor, msk_tensor, seg_tensor)
        probs = probs.tolist()
        assert len(probs) == len(evi_list)
        for i in range(len(probs)):
            if ids[i] not in all_predict:
                all_predict[ids[i]] = []
            #if probs[i][1] >= probs[i][0]:
            all_predict[ids[i]].append(evi_list[i] + [probs[i]])
    return all_predict




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', help='train path')
    parser.add_argument('--name', help='train path')
    parser.add_argument("--batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument('--outdir', required=True, help='path to output directory')
    parser.add_argument('--bert_pretrain', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--evi_num", type=int, default=5, help='Evidence num.')
    parser.add_argument("--threshold", type=float, default=0.0, help='Evidence num.')
    parser.add_argument("--max_len", default=120, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    handlers = [logging.FileHandler(os.path.abspath(args.outdir) + '/train_log.txt'), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)
    logger.info('Start training!')

    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain, do_lower_case=False)
    logger.info("loading training set")
    validset_reader = DataLoaderTest(args.test_path, tokenizer, args, batch_size=args.batch_size)

    logger.info('initializing estimator model')
    bert_model = BertForSequenceEncoder.from_pretrained(args.bert_pretrain)
    bert_model = bert_model.cuda()
    model = inference_model(bert_model, args)
    model.load_state_dict(torch.load(args.checkpoint)['model'])
    model = model.cuda()
    logger.info('Start eval!')
    save_path = args.outdir + "/" + args.name
    predict_dict = eval_model(model, validset_reader)
    save_to_file(predict_dict, save_path)
import numpy as np
import torch
from torch import nn
import json
import jsonlines
from torch.autograd import Variable


def bert_sentence_pair_tokenzier(q_tokens, p_tokens, tokenizer, max_seq_length):
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    tokens = tokens + q_tokens
    segment_ids = segment_ids + [0] * len(q_tokens)
    tokens.append("[SEP]")
    segment_ids.append(0)
    tokens = tokens + p_tokens
    segment_ids = segment_ids + [1] * len(p_tokens)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    input_ids = input_ids + [0] * (max_seq_length - len(input_ids))
    input_mask = input_mask + [0] * (max_seq_length - len(input_mask))
    segment_ids = segment_ids + [0] * (max_seq_length - len(segment_ids))

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids


class BertDataLoaderDev(object):
    ''' For data iteration '''

    def __init__(self,tokenizer, args, batch_size=64):
        self.max_query_len = args.max_query_len
        self.max_seq_len = args.max_seq_len
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.data = self.read_file(args.corpus, args.abstract_retrieval, args.dataset)
        self.total_num = len(self.data)
        self.total_step = np.ceil(self.total_num * 1.0 / batch_size)
        self.step = 0


    def read_file(self, corpus, abstract_retrieval, dataset):
        all_data = []
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
        abstract_retrieval = jsonlines.open(abstract_retrieval)
        dataset = jsonlines.open(dataset)
        for data, retrieval in list(zip(dataset, abstract_retrieval)):
            assert data['id'] == retrieval['claim_id']
            claim = data['claim']
            for doc_id in retrieval['doc_ids']:
                doc = corpus[doc_id]
                sentences = doc['abstract']
                title = doc["title"]
                abstract = " ".join(sentences)
                abstract = title + " " + abstract
                query_toks = self.tokenizer.tokenize(claim)
                doc_toks = self.tokenizer.tokenize(abstract)

                all_data.append({
                    'query_id': retrieval['claim_id'],
                    'doc_id': doc_id,
                    'query_toks': query_toks,
                    'doc_toks': doc_toks})
        return all_data


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        if self.step < self.total_step:
            query_ids, doc_ids, input_ids, input_masks, segment_ids = [], [], [], [], []
            for i in range(self.step * self.batch_size, min((self.step + 1) * self.batch_size, self.total_num)):
                query_ids.append(self.data[i]['query_id'])
                doc_ids.append(self.data[i]['doc_id'])

                query_toks = self.data[i]['query_toks'][:self.max_query_len]
                doc_toks = self.data[i]['doc_toks'][:self.max_seq_len]

                max_len = self.max_seq_len + self.max_query_len + 3
                input_id, input_mask, segment_id = bert_sentence_pair_tokenzier(query_toks, doc_toks, self.tokenizer, max_len)

                input_ids.append(input_id)
                input_masks.append(input_mask)
                segment_ids.append(segment_id)

            input_ids = Variable(
                torch.LongTensor(input_ids)).cuda()
            input_masks = Variable(
                torch.LongTensor(input_masks)).cuda()
            segment_ids = Variable(
                torch.LongTensor(segment_ids)).cuda()
            self.step += 1
            return (query_ids, doc_ids, input_ids, input_masks, segment_ids)
        else:
            self.step = 0
            raise StopIteration()






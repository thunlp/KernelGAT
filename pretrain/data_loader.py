import os
import torch
import numpy as np
import json
from torch.autograd import Variable


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def tok2int_sent(sentence, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""
    sent_a, sent_b = sentence
    tokens_a = tokenizer.tokenize(sent_a)

    tokens_b = None
    if sent_b:
        tokens_b = tokenizer.tokenize(sent_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens =  ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    if tokens_b:
        tokens = tokens + tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_seq_length - len(input_ids))

    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids





def tok2int_list(src_list, tokenizer, max_seq_length, max_seq_size=-1):
    inp_padding = list()
    msk_padding = list()
    seg_padding = list()
    for step, sent in enumerate(src_list):
        input_ids, input_mask, input_seg = tok2int_sent(sent, tokenizer, max_seq_length)
        inp_padding.append(input_ids)
        msk_padding.append(input_mask)
        seg_padding.append(input_seg)
    #if max_seq_size != -1:
    #    inp_padding = inp_padding[:max_seq_size]
    #    msk_padding = msk_padding[:max_seq_size]
    #    seg_padding = seg_padding[:max_seq_size]
    #    inp_padding += ([[0] * max_seq_length] * (max_seq_size - len(inp_padding)))
    #    msk_padding += ([[0] * max_seq_length] * (max_seq_size - len(msk_padding)))
    #    seg_padding += ([[0] * max_seq_length] * (max_seq_size - len(seg_padding)))
    return inp_padding, msk_padding, seg_padding


class DataLoader(object):
    ''' For data iteration '''

    def __init__(self, data_path, tokenizer, args, test=False, cuda=True, batch_size=64):
        self.cuda = cuda

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.evi_num = args.evi_num
        self.threshold = args.threshold
        self.data_path = data_path
        self.test = test
        examples = self.read_file(data_path)
        self.examples = examples
        self.total_num = len(examples)
        if self.test:
            self.total_num = len(self.examples)
            self.total_step = np.ceil(self.total_num * 1.0 / batch_size)
        else:
            self.total_step = self.total_num / batch_size
            self.shuffle()
        self.step = 0



    def read_file(self, data_path):
        examples = list()
        with open(data_path) as fin:
            for step, line in enumerate(fin):
                data = json.loads(line)
                claim = data["claim"]
                evidences = data["evidence"]
                pos_evi = list()
                neg_evi = list()
                for evidence in evidences:
                    if (evidence[3] == 1 or evidence[3] == 2) and evidence[2].strip() != "":
                        pos_evi.append(evidence)
                    elif evidence[3] == 0  and evidence[2].strip() != "":
                        neg_evi.append(evidence)
                total_triples = pos_evi
                pos_num = len(pos_evi)
                neg_num = self.evi_num * pos_num
                np.random.shuffle(neg_evi)
                neg_evi = neg_evi[:neg_num]
                total_triples += neg_evi
                for triple in total_triples:
                    if triple[3] == 1 or triple[3] == 2:
                        examples.append([claim, triple[2], 1])
                    elif triple[3] == 0:
                        examples.append([claim, triple[2], 0])
        return examples


    def shuffle(self):
        np.random.shuffle(self.examples)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''
        if self.step < self.total_step:
            examples = self.examples[self.step * self.batch_size : (self.step+1)*self.batch_size]
            inputs = list()
            labels = list()
            for example in examples:
                inputs.append([example[0], example[1]])
                labels.append(example[2])
            inp, msk, seg = tok2int_list(inputs, self.tokenizer, self.max_len)

            inp_tensor = Variable(
                torch.LongTensor(inp))
            msk_tensor = Variable(
                torch.LongTensor(msk))
            seg_tensor = Variable(
                torch.LongTensor(seg))
            lab_tensor = Variable(
                torch.LongTensor(labels))
            if self.cuda:
                inp_tensor = inp_tensor.cuda()
                msk_tensor = msk_tensor.cuda()
                seg_tensor = seg_tensor.cuda()
                lab_tensor = lab_tensor.cuda()
            self.step += 1
            return inp_tensor, msk_tensor, seg_tensor, lab_tensor
        else:
            self.step = 0
            if not self.test:
                examples = self.read_file(self.data_path)
                self.examples = examples
                self.shuffle()
            raise StopIteration()

class DataLoaderTest(object):
    ''' For data iteration '''

    def __init__(self, data_path, tokenizer, args, cuda=True, batch_size=64):
        self.cuda = cuda

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.evi_num = args.evi_num
        self.threshold = args.threshold
        self.data_path = data_path
        inputs, ids, evi_list = self.read_file(data_path)
        self.inputs = inputs
        self.ids = ids
        self.evi_list = evi_list

        self.total_num = len(inputs)
        self.total_step = np.ceil(self.total_num * 1.0 / batch_size)
        self.step = 0



    def read_file(self, data_path):
        inputs = list()
        ids = list()
        evi_list = list()
        with open(data_path) as fin:
            for step, line in enumerate(fin):
                instance = json.loads(line.strip())
                claim = instance['claim']
                id = instance['id']
                for evidence in instance['evidence']:
                    ids.append(id)
                    inputs.append([claim, evidence[2]])
                    evi_list.append(evidence)
        return inputs, ids, evi_list


    def shuffle(self):
        np.random.shuffle(self.examples)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''
        if self.step < self.total_step:
            inputs = self.inputs[self.step * self.batch_size : (self.step+1)*self.batch_size]
            ids = self.ids[self.step * self.batch_size: (self.step + 1) * self.batch_size]
            evi_list = self.evi_list[self.step * self.batch_size: (self.step + 1) * self.batch_size]
            inp, msk, seg = tok2int_list(inputs, self.tokenizer, self.max_len, -1)
            inp_tensor_input = Variable(
                torch.LongTensor(inp))
            msk_tensor_input = Variable(
                torch.LongTensor(msk))
            seg_tensor_input = Variable(
                torch.LongTensor(seg))
            if self.cuda:
                inp_tensor_input = inp_tensor_input.cuda()
                msk_tensor_input = msk_tensor_input.cuda()
                seg_tensor_input = seg_tensor_input.cuda()
            self.step += 1
            return inp_tensor_input, msk_tensor_input, seg_tensor_input, ids, evi_list
        else:
            self.step = 0
            raise StopIteration()

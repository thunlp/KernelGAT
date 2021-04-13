import argparse
import jsonlines

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--abstract-retrieval', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--threshold', type=float, default=0.5, required=False)
parser.add_argument('--output-flex', type=str)
parser.add_argument('--output-k2', type=str)
parser.add_argument('--output-k3', type=str)
parser.add_argument('--output-k4', type=str)
parser.add_argument('--output-k5', type=str)
args = parser.parse_args()

corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus)}
dataset = jsonlines.open(args.dataset)
abstract_retrieval = jsonlines.open(args.abstract_retrieval)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device "{device}"')

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()

results = []

with torch.no_grad():
    for data, retrieval in tqdm(list(zip(dataset, abstract_retrieval))):
        assert data['id'] == retrieval['claim_id']
        claim = data['claim']

        evidence_scores = {}
        for doc_id in retrieval['doc_ids']:
            doc = corpus[doc_id]
            sentences = doc['abstract']

            encoded_dict = tokenizer.batch_encode_plus(
                zip(sentences, [claim] * len(sentences)),
                pad_to_max_length=True,
                return_tensors='pt'
            )
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            sentence_scores = torch.softmax(model(**encoded_dict)[0], dim=1)[:, 1].detach().cpu().numpy()
            evidence_scores[doc_id] = sentence_scores
        results.append({
            'claim_id': retrieval['claim_id'],
            'evidence_scores': evidence_scores
        })


def output_k(output_path, k=None):
    output = jsonlines.open(output_path, 'w')
    for result in results:
        if k:
            evidence = {doc_id: list(sorted(sentence_scores.argsort()[-k:][::-1].tolist()))
                        for doc_id, sentence_scores in result['evidence_scores'].items()}
        else:
            evidence = {}
            for doc_id, sentence_scores in result['evidence_scores'].items():
                score_list = list()
                evidence_list = list()
                for step, score in enumerate(sentence_scores):
                    if score > args.threshold:
                        score_list.append([step, score])
                score_list = sorted(score_list, key=lambda x:x[1], reverse=True)
                for id, score in score_list[:3]:
                    evidence_list.append(id)
                evidence[doc_id] = evidence_list

        output.write({
            'claim_id': result['claim_id'],
            'evidence': evidence
        })



if args.output_flex:
    output_k(args.output_flex)

if args.output_k2:
    output_k(args.output_k2, 2)

if args.output_k3:
    output_k(args.output_k3, 3)

if args.output_k4:
    output_k(args.output_k4, 4)

if args.output_k5:
    output_k(args.output_k5, 5)

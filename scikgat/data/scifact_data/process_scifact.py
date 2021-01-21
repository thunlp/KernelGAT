import jsonlines
import json
corpus = {doc['doc_id']: doc for doc in jsonlines.open("../corpus.jsonl")}
evidence_retrieval = jsonlines.open("../../prediction/rationale_selection_dev_roberta_large.jsonl")
dataset = jsonlines.open("../claims_dev.jsonl")
with open("scifact_dev.jsonl", "w") as fout:
    for data, retrieval in list(zip(dataset, evidence_retrieval)):
        assert data['id'] == retrieval['claim_id']
        for did, sids in retrieval["evidence"].items():
            evidence_list = list()
            label = "NOT_ENOUGH_INFO"
            flag = False
            content = corpus[int(did)]
            title = content["title"]
            golden_sids = dict()
            if did in data["evidence"]:
                for sentence in data["evidence"][did]:
                    label = sentence["label"]
                    for sid in sentence["sentences"]:
                        golden_sids[sid] = 1
            for sid in sids:
                evidence = content["abstract"][sid]
                evidence_list.append([int(did), title, sid, evidence, 0])
                if sid in golden_sids:
                    flag = True
            if flag == False:
                label = "NOT_ENOUGH_INFO"
            fout.write(json.dumps({"id": int(data['id']), "claim": data["claim"], "evidence":evidence_list, "label":label}) + "\n")

evidence_retrieval = jsonlines.open("../../prediction_new/rationale_selection_train_roberta_large.jsonl")
dataset = jsonlines.open("../claims_train.jsonl")
with open("scifact_train.jsonl", "w") as fout:
    for data, retrieval in list(zip(dataset, evidence_retrieval)):
        assert data['id'] == retrieval['claim_id']
        for did, sids in retrieval["evidence"].items():
            evidence_list = list()
            label = "NOT_ENOUGH_INFO"
            flag = False
            content = corpus[int(did)]
            title = content["title"]
            golden_sids = dict()
            if did in data["evidence"]:
                for sentence in data["evidence"][did]:
                    label = sentence["label"]
                    for sid in sentence["sentences"]:
                        golden_sids[sid] = 1
            for sid in sids:
                evidence = content["abstract"][sid]
                evidence_list.append([int(did), title, sid, evidence, 0])
                if sid in golden_sids:
                    flag = True
            if flag == False:
                label = "NOT_ENOUGH_INFO"
            fout.write(json.dumps({"id": int(data['id']), "claim": data["claim"], "evidence":evidence_list, "label":label}) + "\n")

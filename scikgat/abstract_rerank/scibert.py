import torch
import torch.nn as nn
from transformers import *
import torch.nn.functional as F
class SciBertForRanking(nn.Module):
    def __init__(self):
        super(SciBertForRanking, self).__init__()

        self.bert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.classifier = nn.Linear(768, 2)
        #self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, raw_score=None):
        _, features = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        score = self.classifier(features).squeeze(-1)
        score = F.softmax(score, -1)
        if raw_score is not None:
            features = torch.cat([features, raw_score.unsqueeze(1)], 1)
        return score, features

from transformers import BertModel, RobertaModel, GPT2Model, OpenAIGPTModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import torch
import math

def acc(model, x, y):
    return (model.predict(x) != y).mean()

def get_sinusoid(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * \
               (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def evaluate(x):
    x = x[:512]
    r = np.random.permutation(512**2)
    y1 = r[:x.shape[0]] // 512
    y2 = r[:x.shape[0]] % 512
    x1 = x[y1]
    x2 = x[y2]
    x = np.hstack((x1, x2))
    y = y2 > y1
    model = LogisticRegression(max_iter=1000)
    print(cross_val_score(model, x, y, cv=5, scoring=acc).mean())

evaluate(GPT2Model.from_pretrained('gpt2').wpe.weight.data.numpy())
evaluate(BertModel.from_pretrained('bert-base-uncased').embeddings.position_embeddings.weight.data.numpy())
evaluate(RobertaModel.from_pretrained('roberta-base').embeddings.position_embeddings.weight.data.numpy())
evaluate(get_sinusoid(512, 768).numpy())

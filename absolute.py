from transformers import BertModel, RobertaModel, GPT2Model, OpenAIGPTModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import torch
import math

def mse(model, x, y):
    return (np.abs((np.round(model.predict(x)) - y))).mean()

def get_sinusoid(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * \
               (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


x = GPT2Model.from_pretrained('gpt2').wpe.weight.data.numpy()[:512]
y = np.random.permutation(x.shape[0])
x = x[y]
model = LinearRegression()
print(cross_val_score(model, x, y, cv=5, scoring=mse).mean())

x = BertModel.from_pretrained('bert-base-uncased').embeddings.position_embeddings.weight.data.numpy()
y = np.random.permutation(x.shape[0])
x = x[y]
model = LinearRegression()
print(cross_val_score(model, x, y, cv=5, scoring=mse).mean())

x = RobertaModel.from_pretrained('roberta-base').embeddings.position_embeddings.weight.data.numpy()
y = np.random.permutation(x.shape[0])
x = x[y]
model = LinearRegression()
print(cross_val_score(model, x, y, cv=5, scoring=mse).mean())

x = get_sinusoid(512, 768)
y = np.random.permutation(x.shape[0])
x = x[y]
model = LinearRegression()
print(cross_val_score(model, x, y, cv=5, scoring=mse).mean())

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
import math


def get_sinusoid(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * \
               (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def use_pretrained_pe(new_model, model_type, skip, pool=False):
    if model_type is None:
        return new_model
    print(f'use {model_type} position embedding.')
    if model_type == 'zeros':
        pe = np.zeros((embedding.max_len, embedding.dim))
    elif model_type == 'sinusoid':
        pe = get_sinusoid(
                new_model.bert.config.max_position_embeddings,
                new_model.bert.config.hidden_size)
    else:
        if model_type == 'bert':
            from transformers import BertModel
            model = BertModel.from_pretrained('bert-base-uncased')
            pe = model.embeddings.position_embeddings.weight.data.numpy()
        elif model_type == 'gpt2':
            from transformers import GPT2Model
            model = GPT2Model.from_pretrained('gpt2')
            pe = model.wpe.weight.data.numpy()
        elif model_type == 'roberta':
            from transformers import RobertaModel
            model = RobertaModel.from_pretrained('roberta-base')
            pe = model.embeddings.position_embeddings.weight.data.numpy()
        #pe = PCA(embedding.position_embeddings.embedding_dim).fit_transform(pe)
    if skip:
        pe = pe[:512][::4]
    elif pool:
        pe = pe[:512].view(128, 4, -1).mean(1)
    else:
        pe = pe[:128]
    pe = (pe - pe.mean()) / pe.std() * 0.02
    pe = torch.Tensor(pe)
    try:
        new_model.transformer.wpe.weight.data[:128] = pe
        new_model.transformer.wpe.weight.requires_grad = False
    except:
        new_model.bert.embeddings.position_embeddings.weight.data[:128] = pe
        new_model.bert.embeddings.position_embeddings.weight.requires_grad = False
    return new_model

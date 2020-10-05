import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
import math


def get_sin_pe(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * \
               (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def use_pretrained_pe(embedding, pe_type):
    if pe_type is None or pe_type == 'random':
        return embedding

    print(f'use {pe_type} position embedding.')
    if pe_type == 'zeros':
        pe = np.zeros((embedding.num_embeddings, embedding.embedding_dim))
    elif pe_type == 'sinusoid':
        pe = get_sin_pe(embedding.num_embeddings, embedding.embedding_dim)
    else:
        if pe_type == 'bert':
            from transformers import BertModel
            model = BertModel.from_pretrained('bert-base-uncased')
            pe = model.embeddings.position_embeddings.weight.data.numpy()
        elif pe_type == 'gpt2':
            from transformers import GPT2Model
            model = GPT2Model.from_pretrained('gpt2')
            pe = model.wpe.weight.data.numpy()
        elif pe_type == 'roberta':
            from transformers import RobertaModel
            model = RobertaModel.from_pretrained('roberta-base')
            pe = model.embeddings.position_embeddings.weight.data.numpy()
        pe = PCA(embedding.embedding_dim).fit_transform(pe)
        pe = (pe - pe.mean()) / pe.std()
    pe = torch.Tensor(pe)
    embedding.weight.data[:pe.size(0)] = pe
    return embedding

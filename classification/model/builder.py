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

class TransformerClassifier(nn.Module):
    def __init__(self, embedding, n_class, nhead=8, dim_feedforward=512,
                 dropout=0.1):
        super().__init__()
        d_model = embedding.dim
        self.embedding = embedding
        layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(layer, 1,
                nn.LayerNorm(d_model))
        self.d_model = d_model
        self.fc = nn.Linear(d_model, n_class)

    def forward(self, x, mask=None, label=None):
        emb = self.embedding(x)

        # (N, S, E) -> (S, N, E)
        emb = emb.transpose(0, 1)

        padding_mask = (mask == 0)
        hidden = self.encoder(emb, src_key_padding_mask=padding_mask)

        # (S, N, E) -> (N, S, E)
        hidden = hidden.transpose(0, 1)

        logit = self.fc(hidden[:, -1, :])
        if label is not None:
            loss = F.cross_entropy(logit, label)
            return loss, logit
        else:
            return torch.softmax(logit, dim=-1)


class PositionEmbedding(nn.Module):
    def __init__(self, n_emb, dim=256, dropout=0.1, max_len=128):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.word_emb = nn.Embedding(n_emb, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        max_len = x.size(-1)
        ids = torch.arange(max_len, dtype=torch.long, device=x.device)
        ids = ids.unsqueeze(0).expand(x.size())
        word_emb = self.word_emb(x)
        pos_emb = self.pos_emb(ids)
        emb = word_emb + pos_emb
        return self.dropout(emb)


def use_pretrained_pe(embedding, model_type, max_len):
    print(f'use {model_type} position embedding.')
    if model_type == 'zeros':
        pe = np.zeros((embedding.max_len, embedding.dim))
    elif model_type == 'sinusoid':
        pe = get_sinusoid(embedding.max_len, embedding.dim)
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
        pe = PCA(embedding.dim).fit_transform(pe)[:max_len]
        pe = (pe - pe.mean()) / pe.std()
    pe = torch.Tensor(pe)
    embedding.pos_emb.weight.data = pe
    # embedding.pos_emb.weight.requires_grad = False
    return embedding


def build_model(vocab_size, num_labels, max_len=128, pe=None):
    embedding = PositionEmbedding(vocab_size, max_len=max_len)
    if pe is not None:
        embedding = use_pretrained_pe(embedding, pe, max_len)
    model = TransformerClassifier(embedding, num_labels)
    return model

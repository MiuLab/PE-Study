from box import Box
from dataset import create_dataset
from transformers import BertTokenizer
import numpy as np
import torch
import random
from train import ClassificationTrainer
from model import build_model

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    config = Box.from_yaml(filename='config.yaml')
    set_seed(config.seed)
    train_set, dev_set, test_set, vocab_size, num_labels = create_dataset(
            config.data_path, config.task)
    config['model']['num_labels'] = num_labels
    config['model']['vocab_size'] = vocab_size
    model = build_model(**config.model)
    trainer = ClassificationTrainer(model, config)
    trainer.train(train_set, dev_set, test_set, **config.train)


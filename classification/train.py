import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
from dataset import TransformerIterator
import os
#from datetime import datetime


def get_log_dir(config):
    #current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = str(config.task)
    log_dir += '.' + str(config.model.pe)
    return os.path.join('logs', log_dir)


class Trainer(object):
    def __init__(self, model, config):
        self.model = model
        n_params = sum(p.numel() for p in self.model.parameters())
        print('# Params:', n_params)

        self.optimizer = Adam(
                self.model.parameters(),
                **config.optimizer)
        self.device = config.device
        self.model = self.model.to(self.device)
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.global_steps = 0
        self.logging_steps = config.logging_steps
        self.task = config.task
        self.log_dir = get_log_dir(config)
        self.writer = SummaryWriter(flush_secs=30, log_dir=self.log_dir)
        self.best_score = -1

    def train(self, train_set, dev_set=None, test_set=None, n_epoch=1, batch_size=32):
        self.model.zero_grad()
        train_iter = TransformerIterator(
                train_set, batch_size, shuffle=True, device=self.device)
        dev_iter = TransformerIterator(dev_set, batch_size, device=self.device)
        for epoch in range(1, n_epoch + 1):
            print(f'epoch: {epoch}')
            self.train_epoch(train_iter, dev_iter)

        if test_set is not None:
            test_iter = TransformerIterator(test_set, batch_size, device=self.device)
            self.load(os.path.join(self.log_dir, 'model.pt'))
            self.best_score = self.evaluate(test_iter)
        print('{:.4f}'.format(self.best_score),
                file=open(os.path.join(self.log_dir, 'score'), 'w+'))

    def train_epoch(self, loader):
        pass


    def evaluate(self, loader):
        pass

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
         self.model.load_state_dict(torch.load(path))


class ClassificationTrainer(Trainer):
    def train_epoch(self, train_loader, dev_loader=None):
        for batch in tqdm(train_loader):
            self.model.train()
            loss, _ = self.model(batch.text, batch.mask, batch.label)
            loss /= self.gradient_accumulation_steps
            loss.backward()
            self.writer.add_scalar(f'{self.task}/loss', loss.item(), self.global_steps)
            if self.global_steps % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.model.zero_grad()

            self.global_steps += 1
            if self.global_steps % self.logging_steps == 0:
                if dev_loader is not None:
                    accuracy = self.evaluate(dev_loader)
                    self.writer.add_scalar(f'{self.task}/acc', accuracy, self.global_steps)
                    if accuracy > self.best_score:
                        self.best_score = accuracy
                        self.save(os.path.join(self.log_dir, 'model.pt'))

    def evaluate(self, loader):
        total_loss, total_acc = [], []
        self.model.eval()
        with torch.no_grad():
            predict, labels = [], []
            for batch in loader:
                prob = self.model(batch.text, batch.mask)
                predict.append(torch.argmax(prob, -1))
                labels.append(batch.label)

        predict = torch.cat(predict, 0).cpu().numpy()
        labels = torch.cat(labels, 0).cpu().numpy()
        accuracy = (predict == labels).mean()
        return accuracy



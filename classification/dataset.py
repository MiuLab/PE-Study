from torchtext import datasets, data
import pandas as pd
import pickle
import spacy
import os


class TransformerIterator(data.Iterator):
    r'''
    Create batches with attention mask for transformers
    '''
    def __init__(self, dataset, batch_size, *args, **kwargs):
        super().__init__(dataset, batch_size, *args, **kwargs)
        self.pad_id = dataset.fields['text'].vocab['<pad>']

    def _collate_fn(self, batch):
        mask = (batch.text != self.pad_id).long()
        setattr(batch, 'mask', mask)
        return batch


    def __iter__(self):
        for batch in super().__iter__():
            yield self._collate_fn(batch)


def to_examples(dataframe, fields):
    examples = []
    for example in dataframe[['label', 'sentence']].values:
        example = data.Example.fromlist(example, fields)
        examples.append(example)
    return data.Dataset(examples, fields)


def create_dataset(data_path, task, batch_size=8):

    tokenizer = spacy.load('en')

    def _tokenize(text):
        # use spacy tokenizer
        return [x.text for x in tokenizer(text) if x.text != ' ']


    text_field = data.Field(tokenize=_tokenize, lower=True, batch_first=True)
    label_field = data.Field(sequential=False, use_vocab=False)
    fields = [('label', label_field), ('text', text_field)]

    file_path = os.path.join(data_path, task + '.pkl')
    dataset = pickle.load(open(file_path, 'rb'))
    num_label = int(dataset['label'].max()) + 1
    train_set = to_examples(dataset[dataset['split'] == 'train'], fields)
    dev_set = to_examples(dataset[dataset['split'] == 'dev'], fields)
    test_set = to_examples(dataset[dataset['split'] == 'test'], fields)
    if len(dev_set) == 0:
        train_set, dev_set = train_set.split(0.8)
    if len(test_set) == 0:
        test_set = None

    text_field.build_vocab(train_set)

    return train_set, dev_set, test_set, len(text_field.vocab), num_label

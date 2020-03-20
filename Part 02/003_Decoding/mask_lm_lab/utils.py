import os
import json
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, DataLoader
import jsonlines


class Transformer(nn.Module):
    def __init__(self, vocab_size, max_len, dim=256, num_layers=4, nhead=8, dim_ff=512, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.position_embed = nn.Embedding(max_len, dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Linear(dim, vocab_size)

    def features(self, token_indices):
        pos = torch.arange(len(token_indices), device=token_indices.device).unsqueeze(1)
        x = self.token_embed(token_indices) + self.position_embed(pos)
        x = self.encoder(x)
        return x

    def forward(self, token_indices):
        x = self.features(token_indices)
        x = self.projection(x)
        return x


class Dictionary(object):
    def __init__(self, datasets, include_valid=False):
        self.tokens = []
        self.ids = {}

        # add special tokens
        self.add_token('<s>')
        self.add_token('[M]')
        self.add_token('<pad>')
        self.add_token('<unk>')

        for line in tqdm(datasets['train']):
            for w in line:
                self.add_token(w)

        if include_valid is True:
            for line in tqdm(datasets['valid']):
                for w in line:
                    self.add_token(w)

    def add_token(self, w):
        if w not in self.tokens:
            self.tokens.append(w)
            _w_id = len(self.tokens) - 1
            self.ids[w] = _w_id

    def get_id(self, w):
        return self.ids[w]

    def get_token(self, idx):
        return self.tokens[idx]

    def decode_idx_seq(self, l):
        return [self.tokens[i] for i in l]

    def encode_token_seq(self, l):
        return [self.ids[i] if i in self.ids else self.ids['<unk>'] for i in l]

    def __len__(self):
        return len(self.tokens)


class SequenceDataset(Dataset):
    def __init__(self, list_of_token_lists):
        self.input_tensors = []
        for sample in list_of_token_lists:
            self.input_tensors.append(torch.tensor([sample], dtype=torch.long))

    def __len__(self):
        return len(self.input_tensors)

    def __getitem__(self, idx):
        return self.input_tensors[idx]


def tokenize_dataset(datasets, dictionary):
    tokenized_datasets = {}
    for split, dataset in datasets.items():
        _current_dictified = []
        for l in tqdm(dataset):
            l = ['<s>'] + l + ['<s>']
            encoded_l = dictionary.encode_token_seq(l)
            _current_dictified.append(encoded_l)
        tokenized_datasets[split] = _current_dictified
    return tokenized_datasets


def load_personachat(data_dir='data'):
    import os
    import subprocess
    filename = os.path.join(data_dir, 'personachat_all_sentences_train.jsonl')
    if not os.path.exists(filename):
        os.makedirs(data_dir, exist_ok=True)
        url = "https://nyu.box.com/shared/static/q4nvswb0szelivhgyx87vd1056ttqfyi.jsonl"
        args = ['wget', '-O', filename, url]
        subprocess.call(args)

        url = "https://nyu.box.com/shared/static/8krcizo8sms1m0ppy7uiwfcx4a3l5nsq.jsonl"
        args = ['wget', '-O', os.path.join(data_dir, 'personachat_all_sentences_valid.jsonl'), url]
        subprocess.call(args)

    raw_datasets = {}
    for name in ['train', 'valid']:
        raw_datasets[name] = [x['tokens'] for x in
                              jsonlines.Reader(open(os.path.join(data_dir,
                                                                 'personachat_all_sentences_%s.jsonl' % name)))]

    if os.path.exists(os.path.join(data_dir, 'vocab.pkl')):
        vocab = pickle.load(open(os.path.join(data_dir, 'vocab.pkl'), 'rb'))
    else:
        vocab = Dictionary(raw_datasets, include_valid=False)
        pickle.dump(vocab, open(os.path.join(data_dir, 'vocab.pkl'), 'wb'))

    tokenized_datasets = tokenize_dataset(raw_datasets, vocab)
    datasets = {name: SequenceDataset(ds) for name, ds in tokenized_datasets.items()}
    print("Vocab size: %d" % (len(vocab)))
    return raw_datasets, datasets, vocab


def load_lama_squad(download=True):
    import os
    import subprocess
    filename = os.path.join('data', 'Squad', 'test.jsonl')
    if download:
        url = "https://dl.fbaipublicfiles.com/LAMA/data.zip"
        args = ['wget', url]
        subprocess.call(args)
        args = ['unzip', 'data.zip']
        subprocess.call(args)

    data = [line for line in jsonlines.Reader(open(filename, 'r'))]
    return data

def pad_list_of_tensors(list_of_tensors, pad_token):
    max_length = max([t.size(-1) for t in list_of_tensors])
    padded_list = []
    for t in list_of_tensors:
        padded_tensor = torch.cat(
            [t, torch.tensor([[pad_token] * (max_length - t.size(-1))], dtype=torch.long)], dim=-1)
        padded_list.append(padded_tensor)

    padded_tensor = torch.cat(padded_list, dim=0)
    return padded_tensor


def pad_collate_fn(pad_idx, batch):
    input_list = [s for s in batch]
    input_tensor = pad_list_of_tensors(input_list, pad_idx)
    input_tensor = input_tensor.transpose(0, 1)
    return input_tensor


def save(options, stats, model, save_dir, name, best, log=True):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    name = "%s%s.pt" % (name, '_best' if best else '')
    torch.save({
        'options': options,
        'stats': stats,
        'model_dict': model.state_dict()
    }, os.path.join(save_dir, name))
    if log:
        print("Model saved: %s" % name)

def load(save_dir, name, best):
    path = os.path.join(save_dir, "%s%s.pt" % (name, '_best' if best else ''))
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    return checkpoint

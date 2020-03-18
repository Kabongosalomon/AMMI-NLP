import os
import jsonlines
import torch

from tqdm import tqdm
from torch.utils.data import Dataset


class Dictionary(object):
    def __init__(
        self, datasets, include_valid=False, special_tokens=('<bos>', '<eos>', '<pad>', '<unk>')
    ):
        self.tokens = []
        self.ids = {}

        for token in special_tokens:
            self.add_token(token)

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
        if w not in self.ids:
            return self.ids['<unk>']
        return self.ids[w]

    def get_token(self, idx):
        return self.tokens[idx]

    def decode_idx_seq(self, l):
        return [self.tokens[i] for i in l]

    def encode_token_seq(self, l):
        return [self.ids[i] if i in self.ids else self.ids['<unk>'] for i in l]

    def __len__(self):
        return len(self.tokens)


class NgramDataset(Dataset):
    def __init__(self, ngram_dataset):
        super().__init__()
        self.ngrams = [torch.tensor(i, dtype=torch.long) for i in ngram_dataset]

    def __getitem__(self, i):
        sample = self.ngrams[i]
        return sample

    def __len__(self):
        return len(self.ngrams)

def load_personachat(basedir):
    datasets_fnames = {
        'train': os.path.join(basedir, 'personachat_all_sentences_train.jsonl'),
        'valid': os.path.join(basedir, 'personachat_all_sentences_valid.jsonl'),
        'test': os.path.join(basedir, 'personachat_all_sentences_test.jsonl'),
    }
    datasets_text = {
        'train': [],
        'valid': [],
        'test': [],
    }
    for split, fname in datasets_fnames.items():
        for token_dict in jsonlines.open(fname):
            datasets_text[split].append(token_dict['tokens'])
    return datasets_text


# =========== N-gram =========
def batchify(list_minibatch):
    inp_list = [i[:-1] for i in list_minibatch]
    tar_list = [i[-1] for i in list_minibatch]

    inp_tensor = torch.stack(inp_list, dim=0)
    tar_tensor = torch.stack(tar_list, dim=0)

    return inp_tensor, tar_tensor


def tokenize_dataset(datasets, dictionary, ngram_order=2):
    tokenized_datasets = {}
    for split, dataset in datasets.items():
        _current_dictified = []
        for l in dataset:
            l = ['<bos>'] * (ngram_order - 1) + l + ['<eos>']
            encoded_l = dictionary.encode_token_seq(l)
            _current_dictified.append(encoded_l)
        tokenized_datasets[split] = _current_dictified
    return tokenized_datasets


def slice_into_ngrams(tokenized_dataset, ngram_order=5):
    """This function slices the input sequence into ngrams, e.g:
            [0,1,2,3,4,5] with `ngram_order=2` will be sliced into bigrams:
            [0,1], [1,2], [2,3], [3,4], [4,5]."""
    sliced_datasets = {}
    for split, dataset in tokenized_dataset.items():
        _list_of_sliced_ngrams = []
        for seq in dataset:
            ngrams = [seq[i:i+ngram_order] for i in range(len(seq)-ngram_order+1)]
            _list_of_sliced_ngrams.extend(ngrams)
        sliced_datasets[split] = _list_of_sliced_ngrams
    return sliced_datasets


# ======== RNN ===========
class TensoredDataset(Dataset):
    def __init__(self, list_of_lists_of_tokens, pad_token_id):
        self.input_tensors = []
        self.target_tensors = []
        self.pad = pad_token_id

        for sample in list_of_lists_of_tokens:
            self.input_tensors.append(
                torch.tensor([sample[:-1]], dtype=torch.long)
            )
            self.target_tensors.append(
                torch.tensor([sample[1:]], dtype=torch.long)
            )

    def __len__(self):
        return len(self.input_tensors)

    def __getitem__(self, idx):
        return self.input_tensors[idx], self.target_tensors[idx]

    def pad_collate_fn(self, batch):
        input_list = [s[0] for s in batch]
        target_list = [s[1] for s in batch]
        input_tensor = self.pad_list_of_tensors(input_list)
        target_tensor = self.pad_list_of_tensors(target_list)
        return input_tensor, target_tensor

    def pad_list_of_tensors(self, list_of_tensors):
        max_length = max([t.size(-1) for t in list_of_tensors])
        padded_list = []
        for t in list_of_tensors:
            padded_tensor = torch.cat(
                [t, torch.tensor([[self.pad] * (max_length - t.size(-1))], dtype=torch.long)],
                dim=-1
            )
            padded_list.append(padded_tensor)

        padded_tensor = torch.cat(padded_list, dim=0)
        return padded_tensor


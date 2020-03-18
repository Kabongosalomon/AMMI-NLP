import utils
import torch
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


def mask_tokens(inputs, mask_prob, pad_token_id, mask_token_id, vsize):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original."""
    inputs = inputs.clone()
    labels = inputs.clone()
    # Sample tokens in each sequence for masked-LM training
    masked_indices = torch.bernoulli(torch.full(labels.shape, mask_prob)).bool()
    masked_indices = masked_indices & (inputs != pad_token_id)
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = mask_token_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vsize, labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train(datasets, vocab, num_epochs):
    import torch.optim as optim
    from tqdm import tqdm, trange
    from collections import defaultdict


    options = {
        'batch_size': 64,
        'max_len': 65,
        'dim': 256,
        'nhead': 8,
        'lr': 0.001,
        'max_epochs': 50,
        'early_stop': 5,
        'mask_prob': 0.15,
        'print_every': 500
    }

    trainloader = DataLoader(datasets['train'], batch_size=options['batch_size'],
                             collate_fn=lambda x: utils.pad_collate_fn(vocab.get_id('<pad>'), x),
                             shuffle=True)
    validloader = DataLoader(datasets['valid'], batch_size=options['batch_size'],
                             collate_fn=lambda x: utils.pad_collate_fn(vocab.get_id('<pad>'), x),
                             shuffle=False)

    device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')

    model = Transformer(len(vocab), max_len=options['max_len'], dim=options['dim'], nhead=options['nhead']).to(device)

    model_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(model_parameters, lr=options['lr'])

    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)

    stats = defaultdict(list)
    stats['best_valid_loss'] = 10000
    early_stop = 0
    import ipdb; ipdb.set_trace(context=20)
    for epoch in range(options['max_epochs']):
        train_losses = []
        for step, batch in enumerate(trainloader):
            model.train()
            # Mask the batch
            inputs, labels = mask_tokens(batch, mask_prob=options['mask_prob'],
                                         pad_token_id=vocab.get_id('<pad>'),
                                         mask_token_id=vocab.get_id('[M]'),
                                         vsize=len(vocab))
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            logits_ = logits.view(-1, logits.size(2))
            labels_ = labels.view(-1)

            optimizer.zero_grad()
            loss = criterion(logits_, labels_)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            if (step % options['print_every']) == 0:
                avg_loss = sum(train_losses) / len(train_losses)
                print("Epoch %d Step %d\tTrain Loss %.3f" % (epoch, step, avg_loss))
                stats['train_losses'].append(avg_loss)
                train_losses = []


        valid_losses = []
        for batch in validloader:
            model.eval()
            with torch.no_grad():
                # Mask the batch
                inputs, labels = mask_tokens(batch, mask_prob=options['mask_prob'],
                                             pad_token_id=vocab.get_id('<pad>'),
                                             mask_token_id=vocab.get_id('[M]'),
                                             vsize=len(vocab))
                inputs = inputs.to(device)
                labels = labels.to(device)

                logits = model(inputs)
                logits_ = logits.view(-1, logits.size(2))
                labels_ = labels.view(-1)

                loss = criterion(logits_, labels_)
                valid_losses.append(loss.item())

        avg_loss = sum(valid_losses) / len(valid_losses)
        if avg_loss < stats['best_valid_loss']:
            stats['best_valid_loss'] = avg_loss
            utils.save(options, stats, model, 'model', 'model', best=True)
            early_stop = 0
        else:
            early_stop += 1

        stats['valid_loss'].append(loss.item())
        print("=== Epoch %d\tValid Loss %.3f" % (epoch, stats['valid_loss'][-1]))
        if early_stop == options['early_stop']:
            break


if __name__ == '__main__':
    import utils

    raw_datasets, datasets, vocab = utils.load_personachat()
    train(datasets, vocab, 100)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLP Benchmark: Adam vs AdamW on AG_NEWS
=======================================
Purpose: Demonstrate the effectiveness of Decoupled Weight Decay (AdamW) on language models.
Dataset: AG_NEWS (News classification - lighter than IMDB, easy to run on Kaggle)
Model: SimpleLSTM (Embedding -> LSTM -> FC)
"""

import time
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import AG_NEWS
import pandas as pd
import numpy as np

# ============================================================================
# MODEL
# ============================================================================
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        x = self.fc1(embedded)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

# ============================================================================
# UTILS
# ============================================================================
def yield_tokens(data_iter, tokenizer):
    for _, text in data_iter:
        yield tokenizer(text)

def create_data_loaders(batch_size=64):
    print("📦 Downloading/Loading AG_NEWS dataset...")
    train_iter = list(AG_NEWS(split='train'))
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x) - 1

    class AGNewsDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = list(data)
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    train_dataset_full = AGNewsDataset(train_iter)

    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)

    num_train = len(train_dataset_full)
    split_train_ = int(num_train * 0.95)
    split_valid_ = num_train - split_train_

    train_dataset, valid_dataset = random_split(train_dataset_full, [split_train_, split_valid_])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

    return train_dataloader, valid_dataloader, len(vocab)

# ============================================================================
# TRAINING
# ============================================================================
def train(dataloader, model, optimizer, criterion, epoch):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()
    total_loss = 0.0

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        total_loss += loss.item()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)

    return total_loss / len(dataloader), total_acc / total_count

def evaluate(dataloader, model, criterion):
    model.eval()
    total_acc, total_count = 0, 0
    total_loss = 0.0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_loss += loss.item()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_loss / len(dataloader), total_acc / total_count

# ============================================================================
# MAIN
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description='NLP Benchmark Adam vs AdamW')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--results-dir', type=str, default='results_nlp')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    print(f"🚀 NLP Benchmark using device: {device}")

    # Data
    train_loader, valid_loader, vocab_size = create_data_loaders(args.batch_size)
    num_class = 4
    embed_dim = 64
    hidden_dim = 64

    # Experiment configs
    configs = [
        ('Adam', 'L2 Regularization', torch.optim.Adam),
        ('AdamW', 'Decoupled Weight Decay', torch.optim.AdamW)
    ]

    results = []

    for name, desc, opt_cls in configs:
        print(f"\nTesting {name} ({desc})...")
        set_seed(42) # Fixed seed for fair comparison

        model = TextClassificationModel(vocab_size, embed_dim, hidden_dim, num_class).to(device)
        criterion = torch.nn.CrossEntropyLoss()

        # Note: Adam implements weight decay as L2 penalty, AdamW implements it as decoupled decay
        optimizer = opt_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        history = []
        start_t = time.time()

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train(train_loader, model, optimizer, criterion, epoch)
            val_loss, val_acc = evaluate(valid_loader, model, criterion)

            print(f"| Epoch {epoch:3d} | Train Loss {train_loss:5.4f} | Train Acc {train_acc:5.4f} | "
                  f"Val Loss {val_loss:5.4f} | Val Acc {val_acc:5.4f} |")

            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'optimizer': name
            })

        # Save results
        df = pd.DataFrame(history)
        df.to_csv(f"{args.results_dir}/nlp_{name}.csv", index=False)

        results.append({
            'optimizer': name,
            'final_val_acc': history[-1]['val_acc'],
            'time': time.time() - start_t
        })

    print("\n📊 Final NLP Comparison:")
    print(pd.DataFrame(results))

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    main()
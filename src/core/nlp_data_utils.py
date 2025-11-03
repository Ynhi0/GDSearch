"""
NLP Data Utilities for GDSearch

Provides data loaders for text classification tasks:
- IMDB sentiment analysis (binary classification)
- Text preprocessing and tokenization
- Vocabulary building
- Batch collation for variable-length sequences
"""

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from collections import Counter
import numpy as np
from typing import Tuple, List, Dict, Optional
import re


class Vocabulary:
    """Build and manage vocabulary for text data."""
    
    def __init__(self, max_vocab_size: int = 10000, min_freq: int = 2):
        """
        Initialize vocabulary.
        
        Args:
            max_vocab_size: Maximum vocabulary size
            min_freq: Minimum frequency for word inclusion
        """
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_counts = Counter()
        
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts."""
        print(f"Building vocabulary from {len(texts)} texts...")
        
        # Count words
        for text in texts:
            words = simple_tokenize(text)
            self.word_counts.update(words)
        
        # Add words that meet frequency threshold
        vocab_items = [
            (word, count) for word, count in self.word_counts.most_common()
            if count >= self.min_freq
        ]
        
        # Limit vocabulary size
        vocab_items = vocab_items[:self.max_vocab_size - 2]  # -2 for PAD and UNK
        
        for idx, (word, _) in enumerate(vocab_items, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"✓ Vocabulary built: {len(self.word2idx)} words")
        print(f"  Most common: {vocab_items[:10]}")
        
    def encode(self, text: str) -> List[int]:
        """Convert text to list of indices."""
        words = simple_tokenize(text)
        return [self.word2idx.get(word, 1) for word in words]  # 1 = UNK
    
    def decode(self, indices: List[int]) -> str:
        """Convert indices back to text."""
        return ' '.join([self.idx2word.get(idx, '<UNK>') for idx in indices])
    
    def __len__(self):
        return len(self.word2idx)


def simple_tokenize(text: str) -> List[str]:
    """
    Simple tokenization: lowercase, remove special chars, split on whitespace.
    
    Args:
        text: Input text string
        
    Returns:
        List of tokens
    """
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Keep only alphanumeric and basic punctuation
    text = re.sub(r'[^a-z0-9\s\'\-]', ' ', text)
    
    # Split and remove extra whitespace
    tokens = text.split()
    
    return tokens


class IMDBDataset(Dataset):
    """IMDB sentiment dataset."""
    
    def __init__(self, texts: List[str], labels: List[int], vocab: Vocabulary, max_len: int = 256):
        """
        Initialize dataset.
        
        Args:
            texts: List of text strings
            labels: List of sentiment labels (0=negative, 1=positive)
            vocab: Vocabulary object
            max_len: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Encode text to indices
        indices = self.vocab.encode(text)
        
        # Truncate or pad
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        
        return {
            'indices': indices,
            'label': label,
            'length': len(indices)
        }


def collate_batch(batch):
    """
    Collate function for DataLoader.
    Pads sequences to the same length within a batch.
    """
    # Get max length in this batch
    max_len = max([item['length'] for item in batch])
    
    # Pad sequences
    padded_indices = []
    labels = []
    lengths = []
    
    for item in batch:
        indices = item['indices']
        padded = indices + [0] * (max_len - len(indices))  # 0 = PAD
        padded_indices.append(padded)
        labels.append(item['label'])
        lengths.append(item['length'])
    
    return {
        'indices': torch.LongTensor(padded_indices),
        'labels': torch.LongTensor(labels),
        'lengths': torch.LongTensor(lengths)
    }


def get_imdb_loaders(
    batch_size: int = 32,
    max_vocab_size: int = 10000,
    max_len: int = 256,
    train_size: Optional[int] = None,
    test_size: Optional[int] = None,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Vocabulary]:
    """
    Create IMDB train and test data loaders.
    
    Args:
        batch_size: Batch size for data loaders
        max_vocab_size: Maximum vocabulary size
        max_len: Maximum sequence length
        train_size: Number of training samples (None = use all 25000)
        test_size: Number of test samples (None = use all 25000)
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, test_loader, vocabulary)
    """
    print("="*80)
    print("Loading IMDB dataset...")
    print("="*80)
    
    # Load dataset
    dataset = load_dataset('imdb')
    
    # Extract train data
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    
    # Extract test data
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']
    
    # Subsample if requested
    if train_size is not None and train_size < len(train_texts):
        np.random.seed(seed)
        indices = np.random.choice(len(train_texts), train_size, replace=False)
        train_texts = [train_texts[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]
        print(f"Using {train_size} training samples (subsampled)")
    else:
        print(f"Using all {len(train_texts)} training samples")
    
    if test_size is not None and test_size < len(test_texts):
        np.random.seed(seed)
        indices = np.random.choice(len(test_texts), test_size, replace=False)
        test_texts = [test_texts[i] for i in indices]
        test_labels = [test_labels[i] for i in indices]
        print(f"Using {test_size} test samples (subsampled)")
    else:
        print(f"Using all {len(test_texts)} test samples")
    
    # Build vocabulary from training data
    vocab = Vocabulary(max_vocab_size=max_vocab_size)
    vocab.build_vocab(train_texts)
    
    # Create datasets
    train_dataset = IMDBDataset(train_texts, train_labels, vocab, max_len)
    test_dataset = IMDBDataset(test_texts, test_labels, vocab, max_len)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )
    
    print(f"✓ Train loader: {len(train_loader)} batches")
    print(f"✓ Test loader: {len(test_loader)} batches")
    print(f"✓ Vocabulary size: {len(vocab)}")
    print("="*80)
    
    return train_loader, test_loader, vocab


if __name__ == '__main__':
    # Test the data loaders
    print("Testing IMDB data loaders...")
    
    train_loader, test_loader, vocab = get_imdb_loaders(
        batch_size=32,
        max_vocab_size=5000,
        max_len=128,
        train_size=1000,  # Small subset for testing
        test_size=200
    )
    
    # Test a batch
    batch = next(iter(train_loader))
    print("\nSample batch:")
    print(f"  Indices shape: {batch['indices'].shape}")
    print(f"  Labels shape: {batch['labels'].shape}")
    print(f"  Lengths: {batch['lengths'][:5]}")
    print(f"\nFirst sequence (decoded):")
    first_seq = batch['indices'][0].tolist()
    print(f"  {vocab.decode(first_seq[:50])}...")
    print(f"  Label: {batch['labels'][0].item()} ({'positive' if batch['labels'][0] else 'negative'})")
    
    print("\n✓ Data loaders working correctly!")

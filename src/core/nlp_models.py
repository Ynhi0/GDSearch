"""
NLP Models for GDSearch

Simple recurrent neural network models for text classification:
- SimpleRNN: Vanilla RNN for sentiment analysis
- SimpleLSTM: LSTM model for better long-term dependencies
- BiLSTM: Bidirectional LSTM for richer representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SimpleRNN(nn.Module):
    """
    Simple RNN for text classification.
    
    Architecture:
    - Embedding layer
    - Vanilla RNN layer
    - Fully connected output layer
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 128,
        num_classes: int = 2,
        num_layers: int = 1,
        dropout: float = 0.5
    ):
        """
        Initialize SimpleRNN.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_size: Size of RNN hidden state
            num_classes: Number of output classes
            num_layers: Number of RNN layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, lengths=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len]
            lengths: Actual lengths of sequences [batch_size]
            
        Returns:
            Output logits [batch_size, num_classes]
        """
        # Embedding: [batch, seq_len] -> [batch, seq_len, embed_dim]
        embedded = self.embedding(x)
        
        # RNN: [batch, seq_len, embed_dim] -> [batch, seq_len, hidden]
        rnn_out, hidden = self.rnn(embedded)
        
        # Use last hidden state
        # hidden: [num_layers, batch, hidden] -> [batch, hidden]
        last_hidden = hidden[-1]
        
        # Dropout and classification
        dropped = self.dropout(last_hidden)
        logits = self.fc(dropped)
        
        return logits


class SimpleLSTM(nn.Module):
    """
    Simple LSTM for text classification.
    
    Architecture:
    - Embedding layer
    - LSTM layer
    - Fully connected output layer
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 128,
        num_classes: int = 2,
        num_layers: int = 1,
        dropout: float = 0.5
    ):
        """
        Initialize SimpleLSTM.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_size: Size of LSTM hidden state
            num_classes: Number of output classes
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, lengths=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len]
            lengths: Actual lengths of sequences [batch_size]
            
        Returns:
            Output logits [batch_size, num_classes]
        """
        # Embedding: [batch, seq_len] -> [batch, seq_len, embed_dim]
        embedded = self.embedding(x)
        
        # LSTM: [batch, seq_len, embed_dim] -> [batch, seq_len, hidden]
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state
        # hidden: [num_layers, batch, hidden] -> [batch, hidden]
        last_hidden = hidden[-1]
        
        # Dropout and classification
        dropped = self.dropout(last_hidden)
        logits = self.fc(dropped)
        
        return logits


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM for text classification.
    
    Architecture:
    - Embedding layer
    - Bidirectional LSTM layer
    - Fully connected output layer
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 128,
        num_classes: int = 2,
        num_layers: int = 1,
        dropout: float = 0.5
    ):
        """
        Initialize BiLSTM.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_size: Size of LSTM hidden state (per direction)
            num_classes: Number of output classes
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        # *2 because bidirectional
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x, lengths=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len]
            lengths: Actual lengths of sequences [batch_size]
            
        Returns:
            Output logits [batch_size, num_classes]
        """
        # Embedding: [batch, seq_len] -> [batch, seq_len, embed_dim]
        embedded = self.embedding(x)
        
        # BiLSTM: [batch, seq_len, embed_dim] -> [batch, seq_len, hidden*2]
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate forward and backward hidden states
        # hidden: [num_layers*2, batch, hidden] -> [batch, hidden*2]
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Dropout and classification
        dropped = self.dropout(combined)
        logits = self.fc(dropped)
        
        return logits


class TextCNN(nn.Module):
    """
    1D CNN for text classification (Kim 2014).
    
    Uses multiple convolutional filters of different sizes
    to capture n-gram features.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_filters: int = 100,
        filter_sizes: tuple = (3, 4, 5),
        num_classes: int = 2,
        dropout: float = 0.5
    ):
        """
        Initialize TextCNN.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            num_filters: Number of filters per filter size
            filter_sizes: Tuple of filter sizes (e.g., (3,4,5) for trigrams, 4-grams, 5-grams)
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Create conv layers for each filter size
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=fs
            )
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, x, lengths=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len]
            lengths: Actual lengths of sequences (unused)
            
        Returns:
            Output logits [batch_size, num_classes]
        """
        # Embedding: [batch, seq_len] -> [batch, seq_len, embed_dim]
        embedded = self.embedding(x)
        
        # Transpose for conv1d: [batch, embed_dim, seq_len]
        embedded = embedded.permute(0, 2, 1)
        
        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            # Conv: [batch, embed_dim, seq_len] -> [batch, num_filters, seq_len-filter_size+1]
            conv_out = F.relu(conv(embedded))
            # Max pool: [batch, num_filters, seq_len-filter_size+1] -> [batch, num_filters]
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        # Concatenate all pooled outputs
        combined = torch.cat(conv_outputs, dim=1)
        
        # Dropout and classification
        dropped = self.dropout(combined)
        logits = self.fc(dropped)
        
        return logits


if __name__ == '__main__':
    # Test models
    print("Testing NLP models...")
    
    vocab_size = 5000
    batch_size = 16
    seq_len = 50
    
    # Create dummy input
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    lengths = torch.randint(20, seq_len, (batch_size,))
    
    models = {
        'SimpleRNN': SimpleRNN(vocab_size),
        'SimpleLSTM': SimpleLSTM(vocab_size),
        'BiLSTM': BiLSTM(vocab_size),
        'TextCNN': TextCNN(vocab_size)
    }
    
    print(f"\nInput shape: {x.shape}")
    print("="*80)
    
    for name, model in models.items():
        print(f"\n{name}:")
        
        # Forward pass
        output = model(x, lengths)
        print(f"  Output shape: {output.shape}")
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")
        
        # Test backward pass
        loss = output.sum()
        loss.backward()
        print(f"  ✓ Backward pass successful")
    
    print("\n" + "="*80)
    print("✓ All models working correctly!")

"""
Tests for NLP data utilities and models.
"""

import pytest
import torch
import numpy as np
from src.core.nlp_data_utils import (
    Vocabulary, simple_tokenize, IMDBDataset, get_imdb_loaders
)
from src.core.nlp_models import SimpleRNN, SimpleLSTM, BiLSTM, TextCNN
from src.core.pytorch_optimizers import SGDWrapper, AdamWrapper


class TestVocabulary:
    """Test Vocabulary class."""
    
    def test_build_vocabulary(self):
        """Test building vocabulary from texts."""
        texts = [
            "hello world",
            "hello python",
            "world of python"
        ]
        vocab = Vocabulary(max_vocab_size=10, min_freq=1)
        vocab.build_vocab(texts)
        
        # Check special tokens
        assert vocab.word2idx['<PAD>'] == 0
        assert vocab.word2idx['<UNK>'] == 1
        
        # Check words were added
        assert 'hello' in vocab.word2idx
        assert 'world' in vocab.word2idx
        assert 'python' in vocab.word2idx
        
        # Check size
        assert len(vocab) >= 5  # <PAD>, <UNK>, hello, world, python
    
    def test_encode_decode(self):
        """Test encoding and decoding texts."""
        texts = ["hello world", "test case"]
        vocab = Vocabulary(max_vocab_size=10, min_freq=1)
        vocab.build_vocab(texts)
        
        # Encode
        encoded = vocab.encode("hello world")
        assert isinstance(encoded, list)
        assert all(isinstance(idx, int) for idx in encoded)
        
        # Decode
        decoded = vocab.decode(encoded)
        assert isinstance(decoded, str)
        assert "hello" in decoded
        assert "world" in decoded
    
    def test_unknown_words(self):
        """Test handling of unknown words."""
        texts = ["hello world"]
        vocab = Vocabulary(max_vocab_size=10, min_freq=1)
        vocab.build_vocab(texts)
        
        # Encode text with unknown word
        encoded = vocab.encode("hello unknown")
        # "unknown" should be mapped to <UNK> token (index 1)
        assert 1 in encoded


class TestTokenization:
    """Test tokenization function."""
    
    def test_simple_tokenize(self):
        """Test basic tokenization."""
        text = "Hello, World! This is a test."
        tokens = simple_tokenize(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)
        
        # Check lowercase conversion
        assert 'hello' in tokens
        assert 'world' in tokens
    
    def test_empty_text(self):
        """Test tokenization of empty text."""
        tokens = simple_tokenize("")
        assert tokens == []
    
    def test_punctuation_removal(self):
        """Test that punctuation is removed."""
        text = "Hello!!! How are you???"
        tokens = simple_tokenize(text)
        
        # Punctuation should be removed
        assert '!' not in tokens
        assert '?' not in tokens


class TestIMDBDataset:
    """Test IMDBDataset class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        # Mock dataset
        class MockDataset:
            def __init__(self):
                self.data = [
                    {'text': 'great movie loved it', 'label': 1},
                    {'text': 'terrible waste of time', 'label': 0},
                    {'text': 'amazing performance', 'label': 1},
                ]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        return MockDataset()
    
    def test_dataset_creation(self, sample_data):
        """Test creating IMDBDataset."""
        # Extract texts and labels
        texts = [item['text'] for item in sample_data.data]
        labels = [item['label'] for item in sample_data.data]
        
        vocab = Vocabulary(max_vocab_size=100, min_freq=1)
        vocab.build_vocab(texts)
        
        dataset = IMDBDataset(texts, labels, vocab, max_len=20)
        
        assert len(dataset) == 3
        
        # Get first item
        item = dataset[0]
        assert isinstance(item, dict)
        assert 'indices' in item
        assert 'label' in item
        assert 'length' in item
        assert isinstance(item['indices'], list)
        assert isinstance(item['label'], int)


class TestNLPModels:
    """Test NLP model architectures."""
    
    @pytest.fixture
    def model_params(self):
        """Common parameters for models."""
        return {
            'vocab_size': 1000,
            'embedding_dim': 50,
            'hidden_size': 64,
            'num_classes': 2
        }
    
    def test_simple_rnn(self, model_params):
        """Test SimpleRNN model."""
        model = SimpleRNN(**model_params)
        
        # Create dummy input [batch_size, seq_len]
        batch_size, seq_len = 4, 10
        x = torch.randint(0, model_params['vocab_size'], (batch_size, seq_len))
        
        # Forward pass
        output = model(x)
        
        assert output.shape == (batch_size, model_params['num_classes'])
        
        # Check backward pass works
        loss = output.sum()
        loss.backward()
    
    def test_simple_lstm(self, model_params):
        """Test SimpleLSTM model."""
        model = SimpleLSTM(**model_params)
        
        batch_size, seq_len = 4, 10
        x = torch.randint(0, model_params['vocab_size'], (batch_size, seq_len))
        
        output = model(x)
        assert output.shape == (batch_size, model_params['num_classes'])
        
        # Check backward pass
        loss = output.sum()
        loss.backward()
    
    def test_bilstm(self, model_params):
        """Test BiLSTM model."""
        model = BiLSTM(**model_params)
        
        batch_size, seq_len = 4, 10
        x = torch.randint(0, model_params['vocab_size'], (batch_size, seq_len))
        
        output = model(x)
        assert output.shape == (batch_size, model_params['num_classes'])
        
        # Check backward pass
        loss = output.sum()
        loss.backward()
    
    def test_textcnn(self, model_params):
        """Test TextCNN model."""
        # TextCNN doesn't take hidden_size
        params = {
            'vocab_size': model_params['vocab_size'],
            'embedding_dim': model_params['embedding_dim'],
            'num_classes': model_params['num_classes']
        }
        model = TextCNN(**params, num_filters=32, filter_sizes=[3, 4, 5])
        
        batch_size, seq_len = 4, 10
        x = torch.randint(0, model_params['vocab_size'], (batch_size, seq_len))
        
        output = model(x)
        assert output.shape == (batch_size, model_params['num_classes'])
        
        # Check backward pass
        loss = output.sum()
        loss.backward()
    
    def test_model_training_step(self, model_params):
        """Test a complete training step with custom optimizer."""
        model = SimpleLSTM(**model_params)
        
        # Use custom optimizer wrapper
        optimizer = AdamWrapper(model.parameters(), lr=0.001)
        
        # Create dummy batch
        batch_size, seq_len = 4, 10
        x = torch.randint(0, model_params['vocab_size'], (batch_size, seq_len))
        y = torch.randint(0, 2, (batch_size,))
        
        # Forward pass
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Check that loss is a valid number
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestDataLoading:
    """Test data loading functionality."""
    
    @pytest.mark.slow
    def test_get_imdb_loaders(self):
        """Test loading IMDB data (slow test, requires download)."""
        # Use very small subset for testing
        train_loader, test_loader, vocab = get_imdb_loaders(
            batch_size=8,
            max_len=50,
            max_vocab_size=1000,
            train_size=100,  # Very small for testing
            test_size=50
        )
        
        # Check loaders
        assert len(train_loader) > 0
        assert len(test_loader) > 0
        
        # Check vocabulary
        assert len(vocab) > 0
        assert vocab.word2idx['<PAD>'] == 0
        assert vocab.word2idx['<UNK>'] == 1
        
        # Check batch
        batch = next(iter(train_loader))
        assert 'indices' in batch
        assert 'labels' in batch
        assert batch['indices'].shape[0] <= 8  # batch_size
        assert batch['labels'].shape[0] <= 8


class TestOptimizerIntegration:
    """Test custom optimizers work with NLP models."""
    
    def test_sgd_wrapper(self):
        """Test SGD wrapper with simple model."""
        model = torch.nn.Sequential(
            torch.nn.Embedding(100, 10),
            torch.nn.Flatten(),
            torch.nn.Linear(100, 2)
        )
        
        optimizer = SGDWrapper(model.parameters(), lr=0.01)
        
        # Dummy forward/backward
        x = torch.randint(0, 100, (4, 10))
        output = model(x)
        loss = output.sum()
        
        optimizer.zero_grad()
        loss.backward()
        
        # Should not raise error
        optimizer.step()
    
    def test_adam_wrapper(self):
        """Test Adam wrapper with simple model."""
        model = torch.nn.Sequential(
            torch.nn.Embedding(100, 10),
            torch.nn.Flatten(),
            torch.nn.Linear(100, 2)
        )
        
        optimizer = AdamWrapper(model.parameters(), lr=0.001)
        
        # Dummy forward/backward
        x = torch.randint(0, 100, (4, 10))
        output = model(x)
        loss = output.sum()
        
        optimizer.zero_grad()
        loss.backward()
        
        # Should not raise error
        optimizer.step()

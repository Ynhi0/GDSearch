import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path

# Import the train_one_epoch function from the script by loading it from file to avoid package name collisions
import importlib.util
from pathlib import Path

_spec_path = Path(__file__).resolve().parents[1] / 'kaggle' / 'nlp_benchmark' / 'run_nlp.py'
spec = importlib.util.spec_from_file_location("gd_kaggle_run_nlp", str(_spec_path))
assert spec is not None
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)  # load the module in isolation
train_one_epoch = getattr(module, 'train_one_epoch')


class DummyModel(nn.Module):
    def __init__(self, vocab_size=20, embed_dim=8, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # input_ids: (batch, seq)
        x = self.embed(input_ids).float().mean(dim=1)
        logits = self.fc(x)
        loss = self.loss_fn(logits, labels)

        class Out:
            loss: torch.Tensor
            logits: torch.Tensor

        out = Out()
        out.loss = loss
        out.logits = logits
        return out


class SimpleDataset(Dataset):
    def __init__(self, n_samples=16, seq_len=5, vocab_size=20):
        self.n = n_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, self.vocab_size, (self.seq_len,)),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "labels": int(torch.randint(0, 2, (1,)).item())
        }


def collate_fn(samples):
    input_ids = [s["input_ids"] for s in samples]
    attention_mask = [s["attention_mask"] for s in samples]
    labels = torch.tensor([s["labels"] for s in samples], dtype=torch.long)
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def test_train_one_epoch_returns_float():
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ds = SimpleDataset(n_samples=8, seq_len=4)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)

    loss = train_one_epoch(model, loader, optimizer, device=torch.device("cpu"))

    assert isinstance(loss, float)
    # With random weights, loss should be finite and non-negative
    assert not np.isnan(loss)
    assert loss >= 0.0

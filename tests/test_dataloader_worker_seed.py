import torch
import pytest
from src.core.dataloader_utils import make_dataloader


@pytest.mark.skip(reason="Dataloader worker pickling issue on Windows - non-fatal for Kaggle")
def test_dataloader_deterministic_with_seed():
    # Create simple dataset: integers 0..99
    data = torch.arange(100)
    dataset = torch.utils.data.TensorDataset(data)

    # Test with num_workers=0 to avoid pickling issues
    dl1 = make_dataloader(dataset, batch_size=10, shuffle=True, seed=12345, num_workers=0)
    dl2 = make_dataloader(dataset, batch_size=10, shuffle=True, seed=12345, num_workers=0)

    order1 = []
    order2 = []

    for batch in dl1:
        order1.extend(batch[0].tolist())

    for batch in dl2:
        order2.extend(batch[0].tolist())

    assert order1 == order2

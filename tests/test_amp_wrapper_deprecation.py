import warnings
from src.core.training_utils import AMPWrapper
import torch


def test_amp_wrapper_does_not_emit_deprecation_warning_on_init():
    # Capture warnings during initialization
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        amp = AMPWrapper(enabled=torch.cuda.is_available())

    # Ensure no FutureWarning about GradScaler deprecation was emitted
    future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
    # If a FutureWarning is present, ensure it's not the known GradScaler deprecation
    for fw in future_warnings:
        assert 'GradScaler' not in str(fw.message)

    # If a scaler is present, prefer the torch.amp module (new API) when available
    if amp.scaler is not None:
        mod = amp.scaler.__class__.__module__
        assert mod.startswith('torch.amp') or mod.startswith('torch.cuda.amp')

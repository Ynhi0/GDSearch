"""
Optimizer wrappers for simulating distributed training effects.
"""
from collections import deque
from typing import Deque, List, Optional
import torch


class DelayedOptimizer:
    """
    Wrap a torch optimizer to apply gradients with a fixed delay (in steps).

    Mechanism:
      - On each step(), capture current gradients and enqueue them.
      - If the queue has at least delay_steps gradients, dequeue the oldest
        gradients and apply them to the model parameters (overwrite p.grad).
      - Call the underlying optimizer.step() to update parameters using the
        delayed gradients.
      - Finally zero gradients as usual.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, delay_steps: int = 1):
        if delay_steps < 1:
            raise ValueError("delay_steps must be >= 1")
        self.optimizer = optimizer
        self.delay_steps = delay_steps
        # Freeze parameter order for consistent gradient mapping
        self.params: List[torch.nn.Parameter] = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.params.append(p)
        self.grad_queue: Deque[List[Optional[torch.Tensor]]] = deque(maxlen=delay_steps)

    @torch.no_grad()
    def _capture_current_grads(self) -> List[Optional[torch.Tensor]]:
        captured: List[Optional[torch.Tensor]] = []
        for p in self.params:
            if p.grad is None:
                captured.append(None)
            else:
                captured.append(p.grad.detach().clone())
        return captured

    @torch.no_grad()
    def _load_grads(self, grads: List[Optional[torch.Tensor]]):
        for p, g in zip(self.params, grads):
            if g is None:
                p.grad = None
            else:
                if p.grad is None:
                    p.grad = g.clone()
                else:
                    p.grad.copy_(g)

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        # Enqueue current grads
        current_grads = self._capture_current_grads()
        self.grad_queue.append(current_grads)

        if len(self.grad_queue) < self.delay_steps:
            # Not enough history yet, behave like a no-op update (use zero grads)
            # Option: skip update until queue fills.
            # Here we skip parameter update and just zero grads.
            self.zero_grad()
            return

        # Pop the oldest gradients and apply them
        delayed_grads = self.grad_queue.popleft()
        self._load_grads(delayed_grads)

        # Perform the actual optimizer step with delayed gradients
        loss = self.optimizer.step(closure=closure)
        self.zero_grad()
        return loss

    @property
    def param_groups(self):
        return self.optimizer.param_groups

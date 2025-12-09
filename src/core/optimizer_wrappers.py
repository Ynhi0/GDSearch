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
        
        # 🐛 BUG FIX (Dec 2025): Validate gradient shapes
        if len(current_grads) != len(self.params):
            raise ValueError(f"Gradient count mismatch: {len(current_grads)} vs {len(self.params)}")
        
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
    
    def state_dict(self):
        """
        AUDIT FIX: Save complete wrapper state for checkpoint persistence.
        
        Returns dict containing:
        - optimizer: base optimizer state
        - delay_steps: delay configuration
        - grad_queue: queued gradients (as nested lists for serializability)
        - param_count: number of tracked parameters for validation
        """
        # Convert queue of tensor lists to serializable nested lists
        serialized_queue = []
        for grad_snapshot in self.grad_queue:
            snapshot_list = []
            for g in grad_snapshot:
                if g is None:
                    snapshot_list.append(None)
                else:
                    snapshot_list.append(g.cpu().tolist())
            serialized_queue.append(snapshot_list)
        
        return {
            'optimizer': self.optimizer.state_dict(),
            'delay_steps': self.delay_steps,
            'grad_queue': serialized_queue,
            'param_count': len(self.params),
        }
    
    def load_state_dict(self, state_dict):
        """
        AUDIT FIX: Restore complete wrapper state from checkpoint.
        
        Validates param_count matches and reconstructs grad_queue as tensors.
        """
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.delay_steps = state_dict['delay_steps']
        
        # Validate parameter count
        if state_dict['param_count'] != len(self.params):
            raise ValueError(
                f"Parameter count mismatch: checkpoint has {state_dict['param_count']} "
                f"but model has {len(self.params)} parameters"
            )
        
        # Reconstruct grad_queue from serialized lists
        self.grad_queue.clear()
        for snapshot_list in state_dict['grad_queue']:
            grad_snapshot = []
            for i, g_data in enumerate(snapshot_list):
                if g_data is None:
                    grad_snapshot.append(None)
                else:
                    # Reconstruct tensor with same device/dtype as parameter
                    param = self.params[i]
                    grad_tensor = torch.tensor(g_data, dtype=param.dtype, device=param.device)
                    grad_snapshot.append(grad_tensor)
            self.grad_queue.append(grad_snapshot)

"""
OOM-safe training utilities for PyTorch models.

This module provides robust out-of-memory (OOM) handling with automatic batch size
reduction, taint tracking for validity, and proper error recovery.

Validity Note:
When OOM occurs and batch size is reduced, the training run is marked as "tainted"
to indicate that the run used variable batch sizes, which invalidates fair optimizer
comparisons in benchmarking studies.

CRITICAL - SAM Optimizer Compatibility:
SAM (Sharpness-Aware Minimization) and other closure-based optimizers (e.g., L-BFGS)
are INCOMPATIBLE with OOM retry logic. This is because:
1. SAM requires calling optimizer.step(closure) where the closure is called multiple times
2. Retrying with reduced batch size would corrupt the adversarial gradient computation
3. SAM maintains internal state that cannot be safely recovered after OOM

When a closure-based optimizer is detected (requires_closure=True attribute), OOM retry
is DISABLED and the function will fail immediately on OOM. Users must manually reduce
batch size for SAM experiments.

Supported Optimizers:
- Full OOM retry: SGD, Adam, AdamW, RMSProp, Adagrad (standard optimizers)
- No retry (fail-fast): SAM, Lookahead+SAM, L-BFGS (closure-based optimizers)
"""
import logging
from typing import Tuple, Any, Callable
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
import inspect


def _call_optimizer_step(optimizer: Any, closure: Optional[Callable] = None):
    """Call optimizer.step safely, passing a closure if required.

    This helper introspects the optimizer.step signature and decides whether to
    call with a closure argument. It accepts Any to avoid static check issues at
    call sites that can't assume the exact optimizer signature.

    Contract enforcement: If a closure is provided, the optimizer.step(closure)
    SHOULD return a loss (Tensor or numeric). If it returns None, we raise a
    RuntimeError to enforce a clearer contract rather than silently propagate
    a NoneType loss that will later cause confusing errors.
    """
    try:
        sig = inspect.signature(optimizer.step)
        param = sig.parameters.get('closure')
        if param is not None and param.default is inspect._empty:
            # A required 'closure' parameter is present
            if closure is None:
                raise TypeError("Optimizer.step requires a 'closure' argument but none was provided")
            ret = optimizer.step(closure)
            if ret is None:
                raise RuntimeError(
                    "Optimizer.step(closure) returned None. Closure-based optimizers must return the loss when a closure is provided."
                )
            return ret
        else:
            # Optional closure or no closure param
            if closure is not None:
                ret = optimizer.step(closure)
                if ret is None:
                    raise RuntimeError(
                        "Optimizer.step(closure) returned None. Closure-based optimizers must return the loss when a closure is provided."
                    )
                return ret
            return optimizer.step()
    except (TypeError, ValueError):
        # If signature inspection fails (e.g., C-implemented optimizers), fall back
        # to calling with closure when provided, otherwise call without args.
        if closure is not None:
            ret = optimizer.step(closure)
            if ret is None:
                raise RuntimeError(
                    "Optimizer.step(closure) returned None. Closure-based optimizers must return the loss when a closure is provided."
                )
            return ret
        return optimizer.step()


def oom_safe_train_step(
    model: nn.Module,
    optimizer: Any,  # Accept optimizer wrappers and custom optimizer-like objects
    criterion: Callable,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    max_retries: int = 3,
    min_batch_size: int = 1,
    allow_batchnorm_eval_fallback: bool = False
) -> Tuple[float, int, Any, bool]:
    """
    OOM-safe training step with automatic batch size reduction.
    
    When CUDA OOM occurs, this function automatically reduces the batch size
    and retries the training step. The run is marked as "tainted" to indicate
    that it used variable batch sizes, which invalidates fair optimizer
    comparisons.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer instance
        criterion: Loss function
        inputs: Input tensor batch
        targets: Target tensor batch
        device: torch.device
        max_retries: Maximum OOM recovery attempts
        min_batch_size: Minimum batch size before giving up
        allow_batchnorm_eval_fallback: If True, allows fallback path that temporarily
            switches model to eval() when batch size becomes too small for
            BatchNorm training mode (this changes training semantics and taints the run).
            Default: False (safer behaviour; prefer explicit opt-in).
        
    Returns:
        Tuple of (loss_value, actual_batch_size, outputs, tainted)
        - loss_value: Scalar loss value
        - actual_batch_size: Final batch size used
        - outputs: Model outputs
        - tainted: True if OOM recovery reduced batch size (run is invalid)
        
    Raises:
        RuntimeError: If OOM recovery fails after max_retries or batch too small
    """
    # Determine whether optimizer requires a closure using explicit capability flag when present
    requires_closure_attr = getattr(optimizer, 'requires_closure', None)
    if requires_closure_attr is not None:
        is_closure_based = bool(requires_closure_attr)
    else:
        # Default to False; only set to True for known closure-based optimizers or when attribute explicitly present
        is_closure_based = False
        # Heuristic 1: Inspect optimizer.step signature for a 'closure' parameter.
        try:
            import inspect
            sig = inspect.signature(optimizer.step)
            param = sig.parameters.get('closure')
            # If optimizer.step exposes a 'closure' parameter, only raise for optimizers
            # that are likely closure-based (name heuristics: SAM, LBFGS). Many optimizers
            # include an optional `closure=None` param but do not require it (e.g., SGD, Adam).
            if param is not None:
                cls_name = optimizer.__class__.__name__.upper()
                closure_like = any(x in cls_name for x in ('SAM', 'LBFGS', 'L_BFGS', 'L-BFGS'))
                if closure_like:
                    raise AttributeError(
                        "Closure-detect: optimizer.step accepts a 'closure' parameter but the optimizer instance does not declare 'requires_closure'.\n"
                        "Please set optimizer.requires_closure=True (e.g., for SAM/LBFGS) to opt out of OOM retry logic."
                    )
                else:
                    # Optional closure parameter present but optimizer doesn't appear to be SAM/LBFGS.
                    # Treat as benign optional closure and continue.
                    logging.debug("Optimizer.step has optional 'closure' parameter but class '%s' not flagged as closure-based; continuing.", cls_name)
        except AttributeError:
            # Surface the attribute error to callers to force explicit opt-in/opt-out for closure-based optimizers
            raise
        except Exception as e:
            # If signature inspection fails, fall back to conservative LBFGS detection and log the original error
            logging.debug("Optimizer signature inspection failed: %s", e, exc_info=True)
            try:
                is_closure_based = isinstance(optimizer, torch.optim.LBFGS) or optimizer.__class__.__name__.upper().startswith('LBFGS')
            except Exception as e2:
                logging.debug("Fallback LBFGS detection failed: %s", e2, exc_info=True)
                is_closure_based = False
            if is_closure_based:
                # Set attribute to help downstream checks and log a warning
                try:
                    setattr(optimizer, 'requires_closure', True)
            except (AttributeError, TypeError) as e3:
                    logging.debug("Could not set requires_closure attribute: %s", e3, exc_info=True)
                logging.warning("Optimizer appears to be closure-based (LBFGS); setting 'requires_closure=True' for safety.")

            # Extra heuristic: Some SAM wrappers may not advertise closure semantics but include 'SAM' in class name
            try:
                name_upper = optimizer.__class__.__name__.upper()
                if 'SAM' in name_upper and not is_closure_based:
                    is_closure_based = True
                    try:
                        setattr(optimizer, 'requires_closure', True)
                    except (AttributeError, TypeError):
                        logging.debug("Could not set requires_closure attribute for SAM-like optimizer: %s", e4, exc_info=True)
                    logging.warning("Detected SAM-like optimizer (%s): treating as closure-based and disabling OOM retry for safety.", optimizer.__class__.__name__)
            except Exception:
                # Non-fatal; continue
                pass
    current_inputs = inputs
    current_targets = targets
    retries = 0
    original_batch_size = inputs.size(0)
    tainted = False
    
    while retries < max_retries:
        try:
            current_inputs = current_inputs.to(device)
            current_targets = current_targets.to(device)
            
            # Handle closure-based optimizers (SAM, LBFGS, etc.) that require a closure
            if is_closure_based:
                def closure_retry():
                    optimizer.zero_grad()
                    outputs = model(current_inputs)
                    loss = criterion(outputs, current_targets)
                    loss.backward()
                    return loss

                # Use the closure in optimizer.step
                loss = optimizer.step(closure_retry)

                # Validate closure return type
                if loss is None:
                    raise RuntimeError(
                        f"Closure-based optimizer step returned None. Expected loss tensor. "
                        f"Check implementation: {type(optimizer).__name__}"
                    )

                # Get outputs after step (parameters may have been updated)
                with torch.no_grad():
                    outputs = model(current_inputs)

                # Extract scalar loss value with proper type handling
                if isinstance(loss, torch.Tensor):
                    loss_value = float(loss.item())
                elif isinstance(loss, (int, float)):
                    loss_value = float(loss)
                else:
                    raise TypeError(
                        f"Closure-based step returned unexpected type: {type(loss)}. "
                        f"Expected torch.Tensor or numeric scalar."
                    )

                return loss_value, current_inputs.size(0), outputs, tainted
            
            else:
                # Standard optimizer step
                if getattr(optimizer, 'requires_closure', False):
                    # Optimizer requires closure (e.g., SAM, LBFGS). Provide closure to keep behavior consistent
                    def _closure_for_step():
                        optimizer.zero_grad()
                        outputs = model(current_inputs)
                        loss = criterion(outputs, current_targets)
                        loss.backward()
                        return loss
                    loss = optimizer.step(_closure_for_step)
                    if isinstance(loss, torch.Tensor):
                        loss_value = float(loss.item())
                    elif isinstance(loss, (int, float)):
                        loss_value = float(loss)
                    else:
                        raise TypeError(f"Closure-based step returned unexpected type: {type(loss)}")
                    return loss_value, current_inputs.size(0), model(current_inputs), tainted
                else:
                    optimizer.zero_grad()
                    outputs = model(current_inputs)
                    loss = criterion(outputs, current_targets)
                    loss.backward()
                    
                    # Gradient clipping to prevent explosion
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    if getattr(optimizer, 'requires_closure', False):
                        def _closure_for_step():
                            optimizer.zero_grad()
                            outputs = model(current_inputs)
                            loss = criterion(outputs, current_targets)
                            loss.backward()
                            return loss
                        loss = optimizer.step(_closure_for_step)
                        if isinstance(loss, torch.Tensor):
                            loss_value = float(loss.item())
                        elif isinstance(loss, (int, float)):
                            loss_value = float(loss)
                        else:
                            raise TypeError(f"Closure-based step returned unexpected type: {type(loss)}")
                        # Check for loss divergence
                        if not np.isfinite(loss_value):
                            logging.warning("Loss divergence detected: %s", loss_value)
                            return float('inf'), current_inputs.size(0), model(current_inputs), tainted
                        return loss_value, current_inputs.size(0), model(current_inputs), tainted
                    else:
                        _call_optimizer_step(optimizer)
                    
                    # Check for loss divergence
                    if torch.isnan(loss) or torch.isinf(loss):
                        logging.warning("Loss divergence detected: %f", loss.item())
                        return float('inf'), current_inputs.size(0), outputs, tainted
                    
                    return loss.item(), current_inputs.size(0), outputs, tainted
                
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                # CRITICAL: Enforce documented fail-fast for closure-based optimizers
                if is_closure_based:
                    logging.error(
                        "CUDA OOM with closure-based optimizer (e.g., SAM, LBFGS). "
                        "OOM retry is DISABLED for closure-based optimizers to prevent state corruption. "
                        "Please manually reduce batch size."
                    )
                    raise RuntimeError(
                        "OOM with closure-based optimizer. Retry disabled to prevent corruption. "
                        "Reduce batch size manually."
                    ) from e
                
                retries += 1
                torch.cuda.empty_cache()
                
                old_size = current_inputs.size(0)
                new_size = max(min_batch_size, old_size // 2)
                
                # Check BatchNorm compatibility BEFORE reduction
                # Provide more graceful handling for BatchNorm constraints
                if new_size < 2:
                    # New size too small for BatchNorm in training mode
                    if not allow_batchnorm_eval_fallback:
                        logging.error(
                            "OOM: New batch size %d is too small for BatchNorm in training mode and eval-mode fallback is disabled.",
                            new_size
                        )
                        raise RuntimeError("Batch size too small for BatchNorm layers and eval-mode fallback is disabled")

                    # BatchNorm in eval mode uses running stats and doesn't require batch size >= 2
                    try:
                        was_training = model.training
                        model.eval()
                        logging.warning(
                            "CRITICAL: Batch size %d too small for BatchNorm in training mode. "
                            "Temporarily switching to eval mode to avoid crash. "
                            "This run is TAINTED and should be excluded from analysis.",
                            new_size
                        )
                        # Set tainted flag since we're mixing train/eval modes
                        tainted = True
                        
                        # Process the batch in eval mode, then restore
                        current_inputs_small = inputs[:new_size]
                        current_targets_small = targets[:new_size]
                        
                        optimizer.zero_grad(set_to_none=True)
                        current_inputs_small = current_inputs_small.to(device)
                        current_targets_small = current_targets_small.to(device)
                        
                        outputs = model(current_inputs_small)
                        loss = criterion(outputs, current_targets_small)
                        loss.backward()
                        if getattr(optimizer, 'requires_closure', False):
                            def _closure_small():
                                optimizer.zero_grad()
                                outputs = model(current_inputs_small)
                                loss = criterion(outputs, current_targets_small)
                                loss.backward()
                                return loss
                            loss_small = optimizer.step(_closure_small)
                            if isinstance(loss_small, torch.Tensor):
                                loss_small_val = float(loss_small.item())
                            elif isinstance(loss_small, (int, float)):
                                loss_small_val = float(loss_small)
                            else:
                                logging.warning("Closure-based small-step returned unexpected type: %s", type(loss_small))
                                loss_small_val = float('nan')
                            # Restore training mode before returning
                            if was_training:
                                model.train()
                            return loss_small_val, current_inputs_small.size(0), model(current_inputs_small), tainted
                        else:
                            _call_optimizer_step(optimizer)
                        
                        # Restore training mode before returning
                        if was_training:
                            model.train()
                        
                        # Check for loss divergence
                        if torch.isnan(loss) or torch.isinf(loss):
                            logging.warning("Loss divergence detected: %f", loss.item())
                            return float('inf'), new_size, outputs, tainted
                        
                        return loss.item(), new_size, outputs, tainted
                        
                    except Exception as eval_error:
                        # Restore training mode before raising
                        if was_training:
                            model.train()
                        logging.error(
                            "Cannot reduce batch to %d (BatchNorm requires >= 2) "
                            "and failed to process in eval mode: %s",
                            new_size, eval_error
                        )
                        raise RuntimeError(
                            "Batch size too small for BatchNorm layers and eval mode fallback failed"
                        ) from eval_error
                
                if new_size < min_batch_size:
                    logging.error(
                        "OOM: Cannot reduce batch below %d",
                        min_batch_size
                    )
                    raise
                
                # Mark run as tainted and log warning
                tainted = True
                logging.warning(
                    "INTEGRITY WARNING: Run Tainted - "
                    "Batch size reduced from %d to %d due to OOM",
                    original_batch_size,
                    new_size
                )
                logging.warning(
                    "    CUDA OOM! Reducing batch: %d->%d (retry %d/%d)",
                    old_size,
                    new_size,
                    retries,
                    max_retries
                )
                logging.warning(
                    "    This run uses variable batch size and should be "
                    "excluded from fair comparisons."
                )
                
                # Slice the batch
                current_inputs = inputs[:new_size]
                current_targets = targets[:new_size]
                
                # Clear optimizer gradients
                optimizer.zero_grad(set_to_none=True)
            else:
                raise
    
    logging.error("OOM recovery failed after %d retries", max_retries)
    raise RuntimeError(f"CUDA OOM after {max_retries} recovery attempts")


def clear_gpu_memory(force: bool = False):
    """
    Clear GPU memory between experiments to prevent fragmentation and OOM.
    
    This is critical for long-running benchmark suites to:
    - Prevent cumulative memory leaks
    - Avoid fragmentation
    - Ensure consistent performance
    - Prevent OOM crashes
    
    Args:
        force: If True, perform aggressive cleanup
    """
    if torch.cuda.is_available():
        # Synchronize all CUDA streams
        torch.cuda.synchronize()
        
        # Empty the cache
        torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if force:
            # Aggressive cleanup: clear all caches
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
        
        # Log memory state
        try:
            allocated = torch.cuda.memory_allocated() / 1024**2
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            free = total - allocated
            logging.info(
                "GPU memory cleaned: %.1fMB used, %.1fMB free",
                allocated,
                free
            )
        except Exception as e:
            logging.debug("Could not log GPU memory stats: %s", e)

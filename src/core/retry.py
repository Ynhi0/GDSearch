"""Retry utilities with exponential backoff for robust error handling"""

import time
import logging
import random
from typing import Callable, TypeVar, Any, Optional, Tuple, Type
from functools import wraps
import urllib.error

T = TypeVar('T')

# Network-related exceptions that should trigger retries
NETWORK_EXCEPTIONS = (
    urllib.error.URLError,
    OSError,
    RuntimeError,
    ConnectionError,
    TimeoutError,
)


def retry_with_backoff(
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    backoff_factor: float = 2.0,
    max_backoff: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    log_prefix: str = ""
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that retries a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial backoff delay in seconds
        backoff_factor: Multiplier for backoff after each retry
        max_backoff: Maximum backoff delay in seconds
        exceptions: Tuple of exception types to catch and retry
        log_prefix: Prefix for log messages
        
    Returns:
        Decorated function that retries on failure
        
    Example:
        @retry_with_backoff(max_retries=3, initial_backoff=1.0, backoff_factor=2)
        def download_dataset():
            # Code that might fail transiently
            pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            backoff = initial_backoff
            last_exception: Optional[Exception] = None
            
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        prefix = f"{log_prefix} " if log_prefix else ""
                        logging.error(
                            f"{prefix}Failed after {max_retries} attempts: {str(e)}"
                        )
                        raise
                    
                    # Calculate backoff with exponential growth, max cap, and jitter
                    jitter = random.uniform(0, 0.1 * backoff)  # Add up to 10% jitter
                    current_backoff = min(backoff + jitter, max_backoff)
                    prefix = f"{log_prefix} " if log_prefix else ""
                    logging.warning(
                        f"{prefix}Attempt {attempt}/{max_retries} failed: {str(e)}. "
                        f"Retrying in {current_backoff:.1f}s..."
                    )
                    
                    time.sleep(current_backoff)
                    backoff *= backoff_factor
            
            # Should never reach here, but for type safety
            if last_exception:
                raise last_exception
            raise RuntimeError("Retry logic failed unexpectedly")
        
        return wrapper
    return decorator


def retry_operation(
    operation: Callable[..., T],
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    backoff_factor: float = 2.0,
    max_backoff: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    log_prefix: str = "",
    *args: Any,
    **kwargs: Any
) -> T:
    """
    Retry an operation with exponential backoff (functional interface).
    
    This is a functional alternative to the decorator for one-off retry needs.
    
    Args:
        operation: Function to retry
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial backoff delay in seconds
        backoff_factor: Multiplier for backoff after each retry
        max_backoff: Maximum backoff delay in seconds
        exceptions: Tuple of exception types to catch and retry
        log_prefix: Prefix for log messages
        *args: Positional arguments for operation
        **kwargs: Keyword arguments for operation
        
    Returns:
        Result of successful operation
        
    Example:
        result = retry_operation(
            download_file,
            max_retries=3,
            initial_backoff=2.0,
            url="https://example.com/data.zip"
        )
    """
    backoff = initial_backoff
    last_exception: Optional[Exception] = None
    
    for attempt in range(1, max_retries + 1):
        try:
            return operation(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            if attempt == max_retries:
                prefix = f"{log_prefix} " if log_prefix else ""
                logging.error(
                    f"{prefix}Failed after {max_retries} attempts: {str(e)}"
                )
                raise
            
            # Calculate backoff with exponential growth, max cap, and jitter
            jitter = random.uniform(0, 0.1 * backoff)  # Add up to 10% jitter
            current_backoff = min(backoff + jitter, max_backoff)
            prefix = f"{log_prefix} " if log_prefix else ""
            logging.warning(
                f"{prefix}Attempt {attempt}/{max_retries} failed: {str(e)}. "
                f"Retrying in {current_backoff:.1f}s..."
            )
            
            time.sleep(current_backoff)
            backoff *= backoff_factor
    
    # Should never reach here, but for type safety
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic failed unexpectedly")

"""
Parallel experiment runner for multi-GPU systems (e.g., Kaggle T4x2).

This module enables running multiple experiments in parallel across available GPUs,
providing near-linear speedup for experiment batches.

Features:
- Automatic GPU detection and allocation
- Worker process per GPU
- Queue-based task distribution
- Result collection and error handling
- Graceful fallback to sequential execution
"""
import os
import logging
import multiprocessing as mp
import queue
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch


def run_experiment_on_gpu(
    experiment_config: Dict[str, Any],
    gpu_id: int,
    results_dir: Path,
    queue: mp.Queue
) -> None:
    """
    Run single experiment on specific GPU.
    
    This function runs in a separate process with CUDA device set to a specific GPU.
    
    Args:
        experiment_config: Experiment configuration dictionary
        gpu_id: CUDA device ID (0, 1, etc.)
        results_dir: Results directory
        queue: Multiprocessing queue for status updates
    """
    try:
        # Set CUDA device for this process
        # CRITICAL: Must set CUDA_VISIBLE_DEVICES BEFORE any CUDA operations
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # After setting environment, device 0 is the only visible device
        torch.cuda.set_device(0)
        
        logging.info(f"[GPU {gpu_id}] Starting experiment: {experiment_config['name']}")
        
        # Import here to avoid issues with multiprocessing
        from src.experiments.run_nn_experiment import run_experiment
        
        # Run experiment with device explicitly set
        result = run_experiment(
            config=experiment_config,
            device=f'cuda:{gpu_id}',
            results_dir=results_dir
        )
        
        # Send success status
        queue.put({
            'gpu_id': gpu_id,
            'experiment': experiment_config['name'],
            'status': 'success',
            'result': result
        })
        
    except Exception as e:
        logging.error(f"[GPU {gpu_id}] Experiment failed: {e}", exc_info=True)
        queue.put({
            'gpu_id': gpu_id,
            'experiment': experiment_config.get('name', 'unknown'),
            'status': 'error',
            'error': str(e)
        })


class ParallelExperimentRunner:
    """
    Run multiple experiments in parallel across multiple GPUs.
    
    Strategy:
    - Maintains queue of experiments to run
    - Spawns worker process for each available GPU
    - Workers pull experiments from queue and execute
    - Collects results and handles errors
    
    Example:
        runner = ParallelExperimentRunner(num_gpus=2, results_dir=Path('results'))
        
        experiments = [
            {'name': 'exp1', 'model': 'SimpleMLP', 'lr': 0.01, ...},
            {'name': 'exp2', 'model': 'SimpleCNN', 'lr': 0.001, ...},
            ...
        ]
        
        results = runner.run_experiments_parallel(experiments)
    """
    
    def __init__(self, num_gpus: int = 2, results_dir: Path = Path('results'), strict: bool = False):
        """
        Initialize parallel runner.
        
        Args:
            num_gpus: Number of GPUs available for parallel execution
            results_dir: Base directory for experiment results
            strict: If True, raise error when requested GPU count exceeds available
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate GPU availability
        if torch.cuda.is_available():
            actual_gpu_count = torch.cuda.device_count()
            if actual_gpu_count < num_gpus:
                if strict:
                    raise ValueError(
                        f"Requested {num_gpus} GPUs but only {actual_gpu_count} available. "
                        f"Use strict=False to auto-adjust."
                    )
                else:
                    logging.warning(
                        f"Requested {num_gpus} GPUs but only {actual_gpu_count} available. "
                        f"Using {actual_gpu_count} GPUs."
                    )
                    num_gpus = actual_gpu_count
        else:
            logging.warning("CUDA not available. Parallel execution disabled.")
            num_gpus = 0
        
        self.num_gpus = num_gpus
    
    def run_experiments_parallel(
        self,
        experiments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Run experiments in parallel across available GPUs.
        
        Args:
            experiments: List of experiment configuration dictionaries.
                        Each dict must have at minimum: 'name', 'model', 'dataset', 'optimizer', 'lr', 'seed'
            
        Returns:
            List of result dictionaries with keys: 'gpu_id', 'experiment', 'status', 'result'/'error'
        """
        if self.num_gpus < 2:
            logging.warning("Less than 2 GPUs available, falling back to sequential execution")
            return self._run_sequential(experiments)
        
        logging.info(f"Running {len(experiments)} experiments in parallel on {self.num_gpus} GPUs")
        
        # Create experiment queue
        experiment_queue = mp.Queue()
        for exp in experiments:
            experiment_queue.put(exp)
        
        # Create result queue
        result_queue = mp.Queue()
        
        # Spawn worker processes (one per GPU)
        processes = []
        for gpu_id in range(self.num_gpus):
            p = mp.Process(
                target=self._worker,
                args=(gpu_id, experiment_queue, result_queue)
            )
            p.start()
            processes.append(p)
            logging.info(f"Started worker for GPU {gpu_id} (PID: {p.pid})")
        
        # Collect results
        results = []
        completed = 0
        total = len(experiments)
        
        while completed < total:
            result = result_queue.get()
            results.append(result)
            completed += 1
            
            if result['status'] == 'success':
                logging.info(
                    f"✅ [{completed}/{total}] Completed: {result['experiment']} (GPU {result['gpu_id']})"
                )
            else:
                logging.error(
                    f"❌ [{completed}/{total}] Failed: {result['experiment']} (GPU {result['gpu_id']}): {result['error']}"
                )
        
        # Wait for all workers to finish
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                logging.warning(f"Worker process {p.pid} did not terminate cleanly, killing")
                p.terminate()
                p.join()
        
        return results
    
    def _worker(self, gpu_id: int, experiment_queue: mp.Queue, result_queue: mp.Queue) -> None:
        """
        Worker process that pulls experiments from queue and executes them.
        
        Args:
            gpu_id: GPU device ID for this worker
            experiment_queue: Queue of experiments to run
            result_queue: Queue to put results into
        """
        # Set CUDA device for this worker
        if torch.cuda.is_available():
            # CRITICAL: Must set CUDA_VISIBLE_DEVICES BEFORE any CUDA operations
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            # After setting environment, device 0 is the only visible device
            torch.cuda.set_device(0)
        
        while True:
            try:
                # Get next experiment (with timeout to allow worker to exit)
                experiment = experiment_queue.get(timeout=1)
                
                # Run experiment
                run_experiment_on_gpu(experiment, gpu_id, self.results_dir, result_queue)
                
            except queue.Empty:
                # Queue is empty, worker can exit
                break
            except Exception as e:
                logging.error(f"Worker GPU {gpu_id} error: {e}", exc_info=True)
                break
    
    def _run_sequential(self, experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fallback: Run experiments sequentially on single GPU or CPU.
        
        Args:
            experiments: List of experiment configurations
            
        Returns:
            List of results
        """
        results = []
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        from src.experiments.run_nn_experiment import run_experiment
        
        for i, exp_config in enumerate(experiments):
            logging.info(f"[{i+1}/{len(experiments)}] Running (sequential): {exp_config['name']}")
            
            try:
                result = run_experiment(
                    config=exp_config,
                    device=device,
                    results_dir=self.results_dir
                )
                results.append({
                    'gpu_id': 0,
                    'experiment': exp_config['name'],
                    'status': 'success',
                    'result': result
                })
            except Exception as e:
                logging.error(f"Experiment {exp_config['name']} failed: {e}", exc_info=True)
                results.append({
                    'gpu_id': 0,
                    'experiment': exp_config['name'],
                    'status': 'error',
                    'error': str(e)
                })
        
        return results


def detect_gpu_configuration() -> Dict[str, Any]:
    """
    Detect GPU configuration for the system.
    
    Returns:
        Dictionary with GPU information:
        - gpu_count: Number of GPUs available
        - gpu_names: List of GPU names
        - gpu_memory: List of GPU memory sizes (GB)
        - parallel_capable: Whether system can run parallel experiments
        - recommended_parallel: Whether parallel mode is recommended
    """
    if not torch.cuda.is_available():
        return {
            'gpu_count': 0,
            'gpu_names': [],
            'gpu_memory': [],
            'parallel_capable': False,
            'recommended_parallel': False
        }
    
    gpu_count = torch.cuda.device_count()
    gpu_names = []
    gpu_memory = []
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        gpu_names.append(torch.cuda.get_device_name(i))
        gpu_memory.append(props.total_memory / 1024**3)  # Convert to GB
    
    # Parallel mode is capable if 2+ GPUs
    # Recommended if we have 2+ GPUs with similar memory (within 20%)
    parallel_capable = gpu_count >= 2
    recommended_parallel = False
    
    if parallel_capable:
        # Check if GPUs have similar memory (for balanced workload)
        mem_min = min(gpu_memory)
        mem_max = max(gpu_memory)
        memory_balanced = (mem_max - mem_min) / mem_max < 0.2
        recommended_parallel = memory_balanced
    
    return {
        'gpu_count': gpu_count,
        'gpu_names': gpu_names,
        'gpu_memory': gpu_memory,
        'parallel_capable': parallel_capable,
        'recommended_parallel': recommended_parallel
    }

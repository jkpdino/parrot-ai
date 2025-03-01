import torch
import gc
import time
import threading
from typing import List, Dict, Optional
import psutil
import os

class MemoryMonitor:
    """
    A utility class to monitor memory usage during training.
    
    This class can be used to track both GPU and CPU memory usage,
    and provides methods to log memory statistics and detect memory leaks.
    """
    
    def __init__(self, log_interval: int = 10, enabled: bool = True):
        """
        Initialize the memory monitor.
        
        Args:
            log_interval: How often to log memory usage (in seconds)
            enabled: Whether the monitor is enabled
        """
        self.log_interval = log_interval
        self.enabled = enabled
        self.monitoring = False
        self.monitor_thread = None
        self.peak_gpu_memory = 0
        self.peak_cpu_memory = 0
        self.memory_log = []
        
    def start(self):
        """Start monitoring memory usage in a background thread."""
        if not self.enabled:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("Memory monitoring started")
        
    def stop(self):
        """Stop monitoring memory usage."""
        if not self.enabled or not self.monitoring:
            return
            
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("Memory monitoring stopped")
        
    def _monitor_loop(self):
        """Background thread that periodically logs memory usage."""
        while self.monitoring:
            self.log_memory_usage()
            time.sleep(self.log_interval)
            
    def log_memory_usage(self) -> Dict[str, float]:
        """Log current memory usage and return statistics."""
        stats = self.get_memory_stats()
        
        # Update peak values
        if stats.get('gpu_allocated_gb', 0) > self.peak_gpu_memory:
            self.peak_gpu_memory = stats.get('gpu_allocated_gb', 0)
            
        if stats.get('cpu_used_gb', 0) > self.peak_cpu_memory:
            self.peak_cpu_memory = stats.get('cpu_used_gb', 0)
            
        # Add timestamp
        stats['timestamp'] = time.time()
        self.memory_log.append(stats)
        
        # Print current usage
        if torch.cuda.is_available():
            print(f"GPU Memory: {stats['gpu_allocated_gb']:.2f} GB allocated, "
                  f"{stats['gpu_reserved_gb']:.2f} GB reserved, "
                  f"Peak: {self.peak_gpu_memory:.2f} GB")
        
        print(f"CPU Memory: {stats['cpu_used_gb']:.2f} GB used, "
              f"Peak: {self.peak_cpu_memory:.2f} GB")
              
        return stats
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        stats = {}
        
        # Get CPU memory usage
        process = psutil.Process(os.getpid())
        cpu_info = process.memory_info()
        stats['cpu_used_bytes'] = cpu_info.rss
        stats['cpu_used_gb'] = cpu_info.rss / (1024 ** 3)
        
        # Get GPU memory usage if available
        if torch.cuda.is_available():
            stats['gpu_allocated_bytes'] = torch.cuda.memory_allocated()
            stats['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024 ** 3)
            stats['gpu_reserved_bytes'] = torch.cuda.memory_reserved()
            stats['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024 ** 3)
            
        return stats
        
    def detect_leaks(self, threshold_mb: int = 100) -> List[str]:
        """
        Attempt to detect memory leaks by analyzing consecutive memory logs.
        
        Args:
            threshold_mb: Minimum increase in MB to consider as a potential leak
            
        Returns:
            List of strings describing potential memory leaks
        """
        if len(self.memory_log) < 2:
            return ["Not enough data to detect leaks"]
            
        leaks = []
        prev_stats = self.memory_log[0]
        
        for i, stats in enumerate(self.memory_log[1:], 1):
            if torch.cuda.is_available():
                gpu_diff_mb = (stats['gpu_allocated_bytes'] - prev_stats['gpu_allocated_bytes']) / (1024 * 1024)
                if gpu_diff_mb > threshold_mb:
                    leaks.append(f"Potential GPU memory leak detected at log {i}: "
                                f"+{gpu_diff_mb:.2f} MB")
            
            cpu_diff_mb = (stats['cpu_used_bytes'] - prev_stats['cpu_used_bytes']) / (1024 * 1024)
            if cpu_diff_mb > threshold_mb:
                leaks.append(f"Potential CPU memory leak detected at log {i}: "
                            f"+{cpu_diff_mb:.2f} MB")
                
            prev_stats = stats
            
        return leaks
        
    def clear_gpu_cache(self):
        """Clear CUDA cache to free up memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared")
            
    def run_garbage_collection(self):
        """Run Python garbage collection to free up memory."""
        gc.collect()
        print("Garbage collection run")
        
    def get_peak_memory(self) -> Dict[str, float]:
        """Get peak memory usage."""
        return {
            'peak_gpu_gb': self.peak_gpu_memory,
            'peak_cpu_gb': self.peak_cpu_memory
        }
        
    def get_memory_log(self) -> List[Dict[str, float]]:
        """Get the full memory usage log."""
        return self.memory_log
        
    def reset_peaks(self):
        """Reset peak memory usage values."""
        self.peak_gpu_memory = 0
        self.peak_cpu_memory = 0
        print("Peak memory values reset")
        
    def __enter__(self):
        """Context manager support."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support."""
        self.stop() 
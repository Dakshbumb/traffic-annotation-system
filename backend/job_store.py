"""
In-memory job storage with Queue-based signaling.
In production you would replace this with Redis / DB.
"""
from typing import Dict, Optional
from queue import Queue

# In-memory job storage
job_store: Dict[str, dict] = {}

# Signal queue: worker blocks on .get() instead of busy-polling
_job_queue: Queue = Queue()


def save_job(job_id: str, job_data: dict):
    """Save a job and signal the worker that a new job is available."""
    job_store[job_id] = job_data
    _job_queue.put(job_id)


def get_job(job_id: str) -> Optional[dict]:
    return job_store.get(job_id)


def update_job(job_id: str, updates: dict):
    if job_id in job_store:
        job_store[job_id].update(updates)


def wait_for_job(timeout: float = None) -> Optional[str]:
    """Block until a new job is available (or timeout). Returns job_id or None."""
    try:
        return _job_queue.get(timeout=timeout)
    except Exception:
        return None

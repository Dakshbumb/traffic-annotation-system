from typing import Dict

# Simple in-memory job storage.
# In production you would replace this with Redis / DB.
job_store: Dict[str, dict] = {}


def save_job(job_id: str, job_data: dict):
    job_store[job_id] = job_data


def get_job(job_id: str):
    return job_store.get(job_id)


def update_job(job_id: str, updates: dict):
    if job_id in job_store:
        job_store[job_id].update(updates)

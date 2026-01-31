"""
Retry Queue for Failed Operations
- In-memory queue for failed annotation saves
- Background retry with exponential backoff
- Thread-safe operations
"""

import time
import logging
import threading
from queue import Queue, Empty
from typing import List, Callable, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger("retry_queue")


@dataclass
class RetryItem:
    """Item in the retry queue."""
    data: Any
    operation: str
    attempts: int = 0
    max_attempts: int = 3
    created_at: datetime = None
    last_error: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class RetryQueue:
    """
    Thread-safe retry queue with exponential backoff.
    
    Usage:
        queue = RetryQueue()
        queue.start()
        
        # On failure:
        queue.enqueue(annotations, "save_annotations", save_func)
    """
    
    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        max_attempts: int = 3,
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_attempts = max_attempts
        
        self._queue: Queue = Queue()
        self._handlers: dict = {}
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._failed_items: List[RetryItem] = []
        self._lock = threading.Lock()
        
        # Stats
        self.total_enqueued = 0
        self.total_succeeded = 0
        self.total_failed = 0
    
    def register_handler(self, operation: str, handler: Callable):
        """Register a handler function for an operation type."""
        self._handlers[operation] = handler
    
    def enqueue(self, data: Any, operation: str):
        """Add an item to the retry queue."""
        item = RetryItem(
            data=data,
            operation=operation,
            max_attempts=self.max_attempts,
        )
        self._queue.put(item)
        self.total_enqueued += 1
        logger.info(f"Enqueued retry item: {operation} (queue size: {self._queue.qsize()})")
    
    def start(self):
        """Start the background retry thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        logger.info("Retry queue worker started")
    
    def stop(self):
        """Stop the background retry thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info(f"Retry queue stopped. Success: {self.total_succeeded}, Failed: {self.total_failed}")
    
    def _worker_loop(self):
        """Background worker that processes retry items."""
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except Empty:
                continue
            
            self._process_item(item)
    
    def _process_item(self, item: RetryItem):
        """Process a single retry item."""
        handler = self._handlers.get(item.operation)
        if not handler:
            logger.error(f"No handler registered for operation: {item.operation}")
            self._mark_failed(item, "No handler registered")
            return
        
        item.attempts += 1
        delay = min(self.base_delay * (2 ** (item.attempts - 1)), self.max_delay)
        
        try:
            handler(item.data)
            self.total_succeeded += 1
            logger.info(f"Retry succeeded: {item.operation} (attempt {item.attempts})")
        except Exception as e:
            item.last_error = str(e)
            logger.warning(f"Retry failed: {item.operation} (attempt {item.attempts}/{item.max_attempts}): {e}")
            
            if item.attempts < item.max_attempts:
                # Re-queue with delay
                time.sleep(delay)
                self._queue.put(item)
            else:
                self._mark_failed(item, str(e))
    
    def _mark_failed(self, item: RetryItem, error: str):
        """Mark an item as permanently failed."""
        item.last_error = error
        with self._lock:
            self._failed_items.append(item)
        self.total_failed += 1
        logger.error(f"Retry permanently failed: {item.operation} - {error}")
    
    def get_failed_items(self) -> List[RetryItem]:
        """Get list of permanently failed items."""
        with self._lock:
            return list(self._failed_items)
    
    def get_stats(self) -> dict:
        """Get retry queue statistics."""
        return {
            "queue_size": self._queue.qsize(),
            "total_enqueued": self.total_enqueued,
            "total_succeeded": self.total_succeeded,
            "total_failed": self.total_failed,
            "failed_items_count": len(self._failed_items),
        }


def with_retry(
    func: Callable,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """
    Decorator for retrying a function with exponential backoff.
    
    Usage:
        @with_retry(max_attempts=3)
        def my_function():
            ...
    """
    def wrapper(*args, **kwargs):
        last_exception = None
        
        for attempt in range(1, max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                last_exception = e
                if attempt < max_attempts:
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    logger.warning(f"Attempt {attempt}/{max_attempts} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {max_attempts} attempts failed: {e}")
        
        raise last_exception
    
    return wrapper

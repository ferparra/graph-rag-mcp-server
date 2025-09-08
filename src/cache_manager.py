"""
Cache management with TTL and size bounds for improved performance and memory management.
"""

import time
import asyncio
from typing import Any, Dict, Optional, Callable, TypeVar, Generic
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class TTLCache(Generic[T]):
    """Thread-safe cache with time-to-live (TTL) and size limits."""
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 3600,
        name: Optional[str] = None
    ):
        """
        Initialize TTL cache.
        
        Args:
            max_size: Maximum number of items in cache
            ttl_seconds: Time-to-live for cache items in seconds
            name: Optional name for logging
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.name = name or "cache"
        self._cache: OrderedDict[str, tuple[T, float]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    async def get(self, key: str) -> Optional[T]:
        """Get item from cache if it exists and hasn't expired."""
        async with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                if time.time() < expiry:
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return value
                else:
                    # Expired, remove it
                    del self._cache[key]
                    logger.debug(f"{self.name}: Expired cache entry for key {key}")
            
            self._misses += 1
            return None
    
    async def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Set item in cache with TTL."""
        async with self._lock:
            # Use custom TTL or default
            item_ttl = ttl if ttl is not None else self.ttl_seconds
            expiry = time.time() + item_ttl
            
            # Check if we need to evict items
            if key not in self._cache and len(self._cache) >= self.max_size:
                # Evict least recently used item
                evicted_key = next(iter(self._cache))
                del self._cache[evicted_key]
                self._evictions += 1
                logger.debug(f"{self.name}: Evicted {evicted_key} (LRU)")
            
            self._cache[key] = (value, expiry)
            self._cache.move_to_end(key)
    
    async def delete(self, key: str) -> bool:
        """Delete item from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all items from cache."""
        async with self._lock:
            self._cache.clear()
            logger.info(f"{self.name}: Cache cleared")
    
    async def cleanup_expired(self) -> int:
        """Remove expired items from cache. Returns number of items removed."""
        async with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, (_, expiry) in self._cache.items()
                if current_time >= expiry
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                logger.debug(f"{self.name}: Cleaned up {len(expired_keys)} expired items")
            
            return len(expired_keys)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "name": self.name,
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": f"{hit_rate:.2f}%",
                "ttl_seconds": self.ttl_seconds
            }
    
    def __len__(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class AsyncLRUCache:
    """Async-aware LRU cache decorator with size limits."""
    
    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self.cache: OrderedDict[tuple, Any] = OrderedDict()
        self.lock = asyncio.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = (args, tuple(sorted(kwargs.items())))
            
            async with self.lock:
                # Check if in cache
                if key in self.cache:
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    return self.cache[key]
            
            # Call function
            result = await func(*args, **kwargs)
            
            async with self.lock:
                # Add to cache
                self.cache[key] = result
                self.cache.move_to_end(key)
                
                # Evict if over size limit
                if len(self.cache) > self.maxsize:
                    self.cache.popitem(last=False)
            
            return result
        
        # Add cache control methods as attributes
        setattr(wrapper, 'cache_clear', lambda: self.cache.clear())
        setattr(wrapper, 'cache_info', lambda: {
            "size": len(self.cache),
            "maxsize": self.maxsize
        })
        
        return wrapper


class CacheManager:
    """Central cache manager for the application."""
    
    def __init__(self):
        self.caches: Dict[str, TTLCache] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
    
    def create_cache(
        self,
        name: str,
        max_size: int = 1000,
        ttl_seconds: float = 3600
    ) -> TTLCache:
        """Create or get a named cache."""
        if name not in self.caches:
            self.caches[name] = TTLCache(
                max_size=max_size,
                ttl_seconds=ttl_seconds,
                name=name
            )
            logger.info(f"Created cache '{name}' with max_size={max_size}, ttl={ttl_seconds}s")
        return self.caches[name]
    
    def get_cache(self, name: str) -> Optional[TTLCache]:
        """Get a cache by name."""
        return self.caches.get(name)
    
    async def start_cleanup_task(self, interval: float = 300):
        """Start background task to clean up expired cache entries."""
        if self._cleanup_task is not None:
            return  # Already running
        
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(interval)
                    total_cleaned = 0
                    for cache in self.caches.values():
                        cleaned = await cache.cleanup_expired()
                        total_cleaned += cleaned
                    
                    if total_cleaned > 0:
                        logger.info(f"Cache cleanup: removed {total_cleaned} expired items")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in cache cleanup task: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info(f"Started cache cleanup task with {interval}s interval")
    
    async def stop_cleanup_task(self):
        """Stop the background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Stopped cache cleanup task")
    
    async def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        stats = {}
        for name, cache in self.caches.items():
            stats[name] = await cache.get_stats()
        return stats
    
    async def clear_all(self):
        """Clear all caches."""
        for cache in self.caches.values():
            await cache.clear()
        logger.info("All caches cleared")


# Global cache manager instance
cache_manager = CacheManager()


# Convenience decorators
def cached_result(
    cache_name: str = "default",
    ttl: float = 3600,
    key_func: Optional[Callable] = None
):
    """
    Decorator to cache function results.
    
    Args:
        cache_name: Name of cache to use
        ttl: Time-to-live for cached results
        key_func: Optional function to generate cache key from arguments
    """
    def decorator(func: Callable) -> Callable:
        # Get or create cache
        cache = cache_manager.create_cache(cache_name, ttl_seconds=ttl)
        
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Check cache
            cached = await cache.get(cache_key)
            if cached is not None:
                return cached
            
            # Call function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Store in cache
            await cache.set(cache_key, result, ttl)
            return result
        
        def sync_wrapper(*args, **kwargs):
            # For sync functions, run in event loop
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator